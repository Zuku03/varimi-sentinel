package zw.varimi.sentinel

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import org.json.JSONObject

/**
 * VaRimi Sentinel device harness.
 *
 * Loads the three bundled ONNX heads (risk / yield / price) plus the offline
 * lookup snapshot, runs a full advisory on-device, and logs per-advisory
 * latency to Logcat (tag "VaRimiBench") so a true hardware-in-the-loop number
 * can be recorded on a low-end device.
 *
 * Feature layout must match varimi.model.pipeline.FEATURE_ORDER:
 *   [rainfall_mm, ndvi_proxy_0_1, pest_incidents_reported,
 *    irrigation_coverage_pct, input_availability_score_0_100,
 *    province_code, crop_code, season_code]
 * Ordinal codes come from encoder_vocab.json (index in the sorted category
 * list; -1 for unknown), mirroring sklearn's OrdinalEncoder.
 */
class MainActivity : Activity() {

    private data class Row(
        val month: String, val province: String, val district: String,
        val crop: String, val season: String, val numerics: FloatArray,
    )

    private lateinit var env: OrtEnvironment
    private lateinit var riskSession: OrtSession
    private lateinit var yieldSession: OrtSession
    private lateinit var priceSession: OrtSession
    private lateinit var vocab: Map<String, List<String>>
    private lateinit var rows: List<Row>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        env = OrtEnvironment.getEnvironment()
        riskSession = env.createSession(assets.open("risk.onnx").readBytes())
        yieldSession = env.createSession(assets.open("yield.onnx").readBytes())
        priceSession = env.createSession(assets.open("price.onnx").readBytes())
        vocab = loadVocab()
        rows = loadRows()

        val districtSpinner = findViewById<Spinner>(R.id.spinner_district)
        val cropSpinner = findViewById<Spinner>(R.id.spinner_crop)
        val output = findViewById<TextView>(R.id.text_output)

        val districts = rows.map { it.district }.distinct().sorted()
        districtSpinner.adapter =
            ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, districts)
        val crops = rows.map { it.crop }.distinct().sorted()
        cropSpinner.adapter =
            ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, crops)

        findViewById<Button>(R.id.button_advise).setOnClickListener {
            val district = districtSpinner.selectedItem as String
            val crop = cropSpinner.selectedItem as String
            output.text = advise(district, crop)
        }
    }

    private fun loadVocab(): Map<String, List<String>> {
        val json = JSONObject(assets.open("encoder_vocab.json").reader().readText())
        return json.keys().asSequence().associateWith { key ->
            val arr = json.getJSONArray(key)
            (0 until arr.length()).map { arr.getString(it) }
        }
    }

    private fun loadRows(): List<Row> {
        // lookup.csv columns: month,province,district,crop,season,<5 numerics>
        return assets.open("lookup.csv").bufferedReader().readLines()
            .drop(1)
            .filter { it.isNotBlank() }
            .map { line ->
                val p = line.split(",")
                Row(
                    month = p[0], province = p[1], district = p[2],
                    crop = p[3], season = p[4],
                    numerics = FloatArray(5) { i -> p[5 + i].toFloat() },
                )
            }
    }

    private fun code(column: String, value: String): Float =
        (vocab[column]?.indexOf(value) ?: -1).toFloat()

    private fun advise(district: String, crop: String): String {
        val candidates = rows.filter { it.district == district && it.crop == crop }
        if (candidates.isEmpty()) return "No data for $district / $crop."
        val row = candidates.maxByOrNull { it.month }!!

        val features = FloatArray(8)
        row.numerics.copyInto(features, 0)
        features[5] = code("province", row.province)
        features[6] = code("crop", row.crop)
        features[7] = code("season", row.season)

        val t0 = System.nanoTime()
        val input = OnnxTensor.createTensor(env, arrayOf(features))
        val risk = runLabel(riskSession, input)
        val yieldT = runFloat(yieldSession, OnnxTensor.createTensor(env, arrayOf(features)))
        val price = runLabel(priceSession, OnnxTensor.createTensor(env, arrayOf(features)))
        val elapsedMs = (System.nanoTime() - t0) / 1_000_000.0
        Log.i("VaRimiBench", "advisory_ms=%.3f district=%s crop=%s".format(elapsedMs, district, crop))

        return "%s / %s (%s)\nRisk band: %s\nYield outlook: %.2f t/ha\nPrice direction: %s\nLatency: %.1f ms (see Logcat tag VaRimiBench)"
            .format(crop, district, row.month, risk, yieldT, price, elapsedMs)
    }

    @Suppress("UNCHECKED_CAST")
    private fun runLabel(session: OrtSession, input: OnnxTensor): String =
        session.run(mapOf(session.inputNames.first() to input)).use { result ->
            (result[0].value as Array<String>)[0]
        }

    @Suppress("UNCHECKED_CAST")
    private fun runFloat(session: OrtSession, input: OnnxTensor): Float =
        session.run(mapOf(session.inputNames.first() to input)).use { result ->
            (result[0].value as Array<FloatArray>)[0][0]
        }

    override fun onDestroy() {
        riskSession.close(); yieldSession.close(); priceSession.close()
        super.onDestroy()
    }
}