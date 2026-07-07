# Submission Checklist - VaRimi Sentinel (Track 3, AI4I)

## 1. Placeholders to fill (blockers - the PDF ships with these markers)
- [ ] `[LEAD_INNOVATOR_NAME]` - cover page + team table
- [ ] `[TEAM_MEMBER_2..4]` + `[CREDENTIALS]` - team table (2-5 Zimbabwean citizens)
- [ ] `[GIT_URL]` / `[DEMO_URL]` - cover + Section 3 (see step 3)
- [ ] Budget figures - confirm against real quotes (USSD aggregator, hosting,
      stipends); remove the `<!-- REVIEW -->` comments
- [ ] Milestone dates - align to the official AI4I programme calendar
- [ ] ProjectID - assigned by the portal; rebuild PDF with it (step 4)

## 2. Quality gates before publishing
- [ ] `ruff check src tests scripts` clean
- [ ] `pytest -q -p no:warnings` - all tests pass
- [ ] `python -m varimi.model.train && python -m varimi.edge.export &&
      python -m varimi.edge.benchmark` - reports regenerate cleanly
- [ ] Native-speaker sign-off recorded in `docs/TRANSLATION_VALIDATION.md`
      (or keep the "pending validation" language in the proposal - do not
      remove the caveat without the sign-off)
- [ ] Android harness built once in Android Studio; fix any scaffold errors;
      record device latency and update `reports/edge_benchmark.md`

## 3. Publishing (deferred by decision - do before submission)
- [ ] `gh auth login` (once)
- [ ] From `varimi-sentinel/`:
      `gh repo create varimi-sentinel --public --source . --push`
      (repo must be public for free Streamlit hosting)
- [ ] Streamlit Community Cloud (https://share.streamlit.io): New app ->
      repo `varimi-sentinel`, main file `src/varimi/serving/app.py`.
      Add `streamlit` to a root `requirements.txt` first if prompted
      (`pip freeze` the demo extras) and commit.
- [ ] Verify the live demo renders an advisory end-to-end
- [ ] Confirm CI is green on GitHub Actions
- [ ] Replace `[GIT_URL]` / `[DEMO_URL]` in the proposal markdown

## 4. Final PDF
- [ ] `python scripts/build_proposal.py <ProjectID>` - regenerates
      `<ProjectID>_AI4I_Proposal_Development.pdf`
- [ ] Page count <= 10 excluding cover (script warns if over)
- [ ] Typography: Arial/Helvetica 11pt, 1.15 spacing, 1-inch margins (built-in)
- [ ] Open the PDF and visually check tables, the diagram and page breaks
- [ ] No `[PLACEHOLDER]` or `<!-- REVIEW -->` text remains (search the PDF)

## 5. Portal submission (per the Call + Track-3 ToR)
- [ ] Online portal questionnaire completed
- [ ] Git link + demo URL entered in the portal fields
- [ ] Proposal PDF uploaded - **PDF only** (docx is disqualified), filename
      `[ProjectID]_AI4I_Proposal_Development.pdf`
- [ ] Eligibility pack: 2-5 team members, all Zimbabwean citizens; notarised
      affidavit; prior-funding declarations; only one project shortlisted per
      team rule respected
- [ ] Keep a dated copy of everything submitted

## 6. Defense preparation
- [ ] Every team member reads `docs/DEFENSE_QA.md` and can answer Q1-Q12
      without notes
- [ ] Dry-run the demo offline (airplane mode) on the pilot phone