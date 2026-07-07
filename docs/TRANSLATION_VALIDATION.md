# Translation Validation Sheet - Shona (sn) & Ndebele (nd)

**Status: DRAFT - pending native-speaker validation. Do not deploy the sn/nd
strings to real users until every row below is signed off.**

## Instructions for reviewers
You are reviewing the exact strings shown to farmers and extension officers.
For each row: read the English meaning, judge whether the sn/nd rendering is
natural, respectful and unambiguous for a rural farming audience, then mark the
Verdict column (OK / FIX) and write the corrected text if needed. Sign and date
the bottom. Agricultural terms should follow common AGRITEX usage where one
exists.

Known limitation: crop and district names are shown as they appear in the
source data (English), e.g. "Maize", "Mutare".

## 1. Message lead (template)
| Key | English | Shona (draft) | Ndebele (draft) | Verdict | Correction |
|---|---|---|---|---|---|
| lead | {crop} in {district} ({month}): risk {risk}, price {price}. | {crop} muno {district} ({month}): njodzi {risk}, mutengo {price}. | {crop} e-{district} ({month}): ingozi {risk}, intengo {price}. | | |
| advice_prefix | Advice: | Zano: | Iseluleko: | | |

## 2. Risk terms
| Key | English | Shona (draft) | Ndebele (draft) | Verdict | Correction |
|---|---|---|---|---|---|
| risk.Low | Low | Yakaderera | Ephansi | | |
| risk.Medium | Medium | Yepakati | Ephakathi | | |
| risk.High | High | Yakakwirira | Ephezulu | | |

## 3. Price terms
| Key | English | Shona (draft) | Ndebele (draft) | Verdict | Correction |
|---|---|---|---|---|---|
| price.down | falling | ari kudzikira | ehla | | |
| price.flat | stable | akagadzikana | amile | | |
| price.up | rising | ari kukwira | enyuka | | |

## 4. Action phrases
| Key | English | Shona (draft) | Ndebele (draft) | Verdict | Correction |
|---|---|---|---|---|---|
| pests | scout and treat pests; plant resistant varieties where possible | ongororai minda uye rapai zvipembenene; dyarai mbeu dzinodzivirira kana zvichiita | hlolani amasimu lilaphe izinambuzane; hlanyelani inhlanyelo eqinileyo nxa kusenzeka | | |
| irrigation | prioritise irrigation and drought-tolerant inputs | koshesai kudiridza uye mbeu dzinotsungirira kushaya mvura | qakathekisani ukuthelela lenhlanyelo ebekezelela isomiso | | |
| monitor_high | increase field checks and secure inputs early | wedzerai kuongorora minda uye chengetedzai zvinodiwa nekukurumidza | andisani ukuhlola amasimu liqoqe okudingekayo masinyane | | |
| monitor_medium | keep watching conditions and maintain input supply | rambai muchiongorora mamiriro uye chengetedzai zvinodiwa | qhubekani lihlola isimo ligcine okudingekayo | | |
| proceed | conditions look good; proceed with the normal plan | mamiriro akanaka; endererai mberi nehurongwa hwenyu | isimo sihle; qhubekani ngohlelo lwenu | | |
| sell_window | a good market window is forming | nguva yakanaka yekutengesa iri kuuya | isikhathi esihle sokuthengisa siyeza | | |
| hold_sale | think about storage or a later sale | fungai kuchengeta kana kunonoka kutengesa | cabangani ukugcina kumbe ukuphuzisa ukuthengisa | | |

## Sign-off
| Language | Reviewer name | Qualification / affiliation | Date | Signature |
|---|---|---|---|---|
| Shona | | | | |
| Ndebele | | | | |

After corrections are merged into `src/varimi/serving/advisory.py`, update the
Status line at the top of this file to VALIDATED with the sign-off date.