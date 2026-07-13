package zw.varimi.sentinel

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.AutoCompleteTextView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.ChipGroup
import org.json.JSONObject

/**
 * VaRimi Sentinel device harness - Material 3 UI with full advisory parity.
 *
 * Renders the same recommendation as the web (Streamlit) side: localized lead
 * message, driver explanations and recommended action. All user-facing strings
 * come from assets/strings.json, generated from the Python catalogs by
 * scripts/sync_android_assets.py - never hardcode advisory wording here. The
 * small rule layer (actionKeys/drivers) mirrors src/varimi/serving/advisory.py;
 * keep both sides in sync.
 *
 * Feature layout matches varimi.model.pipeline.FEATURE_ORDER:
 *   [rainfall_mm, ndvi_proxy_0_1, pest_incidents_reported,
 *    irrigation_coverage_pct, input_availability_score_0_100,
 *    province_code, crop_code, season_code]
 * Per-advisory latency is logged to Logcat tag "VaRimiBench".
 */
class MainActivity : AppCompatActivity() {

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
    private lateinit var catalog: JSONObject

    private val riskColors = mapOf(
        "Low" to "#2E7D32", "Medium" to "#F9A825", "High" to "#C62828",
    )
    // lookup.csv numeric column index per driver feature (order matters for display)
    private val driverIndex = listOf(
        "pest_incidents_reported" to 2,
        "ndvi_proxy_0_1" to 1,
        "irrigation_coverage_pct" to 3,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        env = OrtEnvironment.getEnvironment()
        riskSession = env.createSession(assets.open("risk.onnx").readBytes())
        yieldSession = env.createSession(assets.open("yield.onnx").readBytes())
        priceSession = env.createSession(assets.open("price.onnx").readBytes())
        vocab = loadVocab()
        rows = loadRows()
        catalog = JSONObject(assets.open("strings.json").reader().readText())

        val districtBox = findViewById<AutoCompleteTextView>(R.id.dropdown_district)
        val cropBox = findViewById<AutoCompleteTextView>(R.id.dropdown_crop)

        val districts = rows.map { it.district }.distinct().sorted()
        districtBox.setAdapter(
            ArrayAdapter(this, android.R.layout.simple_list_item_1, districts)
        )
        districtBox.setText(districts.first(), false)

        fun refreshCrops() {
            val crops = rows.filter { it.district == districtBox.text.toString() }
                .map { it.crop }.distinct().sorted()
            cropBox.setAdapter(ArrayAdapter(this, android.R.layout.simple_list_item_1, crops))
            if (cropBox.text.toString() !in crops) cropBox.setText(crops.first(), false)
        }
        refreshCrops()
        districtBox.setOnItemClickListener { _, _, _, _ -> refreshCrops() }

        findViewById<MaterialButton>(R.id.button_advise).setOnClickListener {
            renderAdvisory(districtBox.text.toString(), cropBox.text.toString())
        }
    }

    // --- asset loading -----------------------------------------------------

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

    // --- localization helpers ----------------------------------------------

    private fun selectedLanguage(): String =
        when (findViewById<ChipGroup>(R.id.chip_group_lang).checkedChipId) {
            R.id.chip_sn -> "sn"
            R.id.chip_nd -> "nd"
            else -> "en"
        }

    private fun lang(section: String): JSONObject =
        catalog.getJSONObject(section)

    // --- rule layer (mirrors src/varimi/serving/advisory.py) ----------------

    private fun drivers(row: Row): List<Triple<String, Double, Boolean>> {
        val thresholds = catalog.getJSONObject("driver_thresholds")
        return driverIndex.map { (feature, idx) ->
            val value = row.numerics[idx]
            val pct = rows.count { it.numerics[idx] <= value }.toDouble() / rows.size
            val cfg = thresholds.getJSONObject(feature)
            val unfavourable = if (cfg.getString("direction") == "high") {
                pct >= cfg.getDouble("cutoff")
            } else {
                pct <= cfg.getDouble("cutoff")
            }
            Triple(feature, pct, unfavourable)
        }
    }

    // Mirrors _action_keys() in src/varimi/serving/advisory.py - keep in sync.
    private fun actionKeys(risk: String, price: String, bad: Set<String>): List<String> {
        val keys = mutableListOf<String>()
        when (risk) {
            "High" -> {
                if ("pest_incidents_reported" in bad) keys.add("pests")
                if ("ndvi_proxy_0_1" in bad || "irrigation_coverage_pct" in bad) {
                    keys.add("irrigation")
                }
                if (keys.isEmpty()) keys.add("monitor_high")
            }
            "Medium" -> keys.add("monitor_medium")
            else -> keys.add("proceed")
        }
        when (price) {
            "up" -> keys.add("sell_window")
            "down" -> keys.add("hold_sale")
        }
        return keys
    }

    // --- inference + rendering ----------------------------------------------

    private fun renderAdvisory(district: String, crop: String) {
        val candidates = rows.filter { it.district == district && it.crop == crop }
        if (candidates.isEmpty()) return
        val row = candidates.maxByOrNull { it.month }!!
        val language = selectedLanguage()

        val features = FloatArray(8)
        row.numerics.copyInto(features, 0)
        features[5] = (vocab["province"]?.indexOf(row.province) ?: -1).toFloat()
        features[6] = (vocab["crop"]?.indexOf(row.crop) ?: -1).toFloat()
        features[7] = (vocab["season"]?.indexOf(row.season) ?: -1).toFloat()

        val t0 = System.nanoTime()
        val risk = runLabel(riskSession, features)
        val yieldT = runFloat(yieldSession, features)
        val price = runLabel(priceSession, features)
        val elapsedMs = (System.nanoTime() - t0) / 1_000_000.0
        Log.i("VaRimiBench", "advisory_ms=%.3f district=%s crop=%s".format(elapsedMs, district, crop))

        val driverList = drivers(row)
        val bad = driverList.filter { it.third }.map { it.first }.toSet()
        val keys = actionKeys(risk, price, bad)

        val riskLocal = lang("risk").getJSONObject(language).getString(risk)
        val priceLocal = lang("price").getJSONObject(language).getString(price)
        val lead = lang("lead").getString(language)
            .replace("{crop}", crop).replace("{district}", district)
            .replace("{month}", row.month).replace("{risk}", riskLocal)
            .replace("{price}", priceLocal)
        val actions = catalog.getJSONObject("actions")
        val action = lang("advice_prefix").getString(language) +
            keys.joinToString("; ") { actions.getJSONObject(it).getString(language) }

        findViewById<View>(R.id.results_container).visibility = View.VISIBLE
        val riskCard = findViewById<MaterialCardView>(R.id.card_risk)
        riskCard.setCardBackgroundColor(Color.parseColor(riskColors[risk] ?: "#607D8B"))
        findViewById<TextView>(R.id.text_risk_band).text = "Risk: $riskLocal ($risk)"
        findViewById<TextView>(R.id.text_metrics).text =
            "Yield outlook: %.2f t/ha    Price: %s".format(yieldT, priceLocal)
        findViewById<TextView>(R.id.text_message).text = lead
        findViewById<TextView>(R.id.text_drivers).text = driverList.joinToString("\n") {
            (feature, pct, unfav) ->
            val name = feature.replace('_', ' ')
            val marker = if (unfav) "  ⚠" else ""
            "%-32s p%02d%s".format(name, (pct * 100).toInt(), marker)
        }
        findViewById<TextView>(R.id.text_action).text = action
        findViewById<TextView>(R.id.text_latency).text =
            "Latency: %.1f ms (Logcat tag VaRimiBench)".format(elapsedMs)
    }

    @Suppress("UNCHECKED_CAST")
    private fun runLabel(session: OrtSession, features: FloatArray): String =
        OnnxTensor.createTensor(env, arrayOf(features)).use { input ->
            session.run(mapOf(session.inputNames.first() to input)).use { result ->
                (result[0].value as Array<String>)[0]
            }
        }

    @Suppress("UNCHECKED_CAST")
    private fun runFloat(session: OrtSession, features: FloatArray): Float =
        OnnxTensor.createTensor(env, arrayOf(features)).use { input ->
            session.run(mapOf(session.inputNames.first() to input)).use { result ->
                (result[0].value as Array<FloatArray>)[0][0]
            }
        }

    override fun onDestroy() {
        riskSession.close(); yieldSession.close(); priceSession.close()
        super.onDestroy()
    }
}