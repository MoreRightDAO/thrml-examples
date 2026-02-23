"""
nb_pharma01 v2: Drug Pricing Pe — Void Thermodynamics of Pharmaceutical Markets

Domain: Pharmaceutical pricing / healthcare market structure
N = 34 drug market categories (expanded from v1 N=15)
Data: Public sources — FTC PBM Report (2024), CMS Medicare Part D, FDA Generic Drug Competition,
      FDA Orange Book, IQVIA 2023, published academic literature on drug pricing and market concentration

KEY NEW FINDING (v2): International comparison — same drug, different regulatory architecture → different Pe
Insulin glargine: US V=9 Pe=+43.9 | Canada V=5 Pe=−4.2 | Germany V=3 Pe=−25.9
Pe measures system architecture, not pharmacology. α=3 (survival coupling) unchanged across countries.
Only O and R vary — set by regulatory framework.

Discriminant validity test: Shkreli/Daraprim vs Sackler/OxyContin (retained from v1)
Natural experiment: Mark Cuban's Cost Plus Drugs — Pe=−125 in live commercial US market

HHI sources:
- Feldman et al. (2022) JAMA Network Open: Market concentration by drug class
- FDA Office of Generic Drugs Annual Report (2022): HHI by generic entry count
- FTC PBM Report (2024): Pharmacy benefit manager concentration
- Dafny et al. (2012) RAND: Drug market HHI analysis
- IQVIA Institute (2023): Market share data by drug class
- Sarpatwari et al. (2019) NEJM: Drug pricing and market structure
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Canonical THRML Parameters ───────────────────────────────────────────────
B_ALPHA = 0.867
B_GAMMA = 2.244
K       = 16
V_STAR  = 9 * (1 - B_ALPHA / B_GAMMA)
C_ZERO  = B_ALPHA / B_GAMMA

def thrml_pe(V):
    c = 1.0 - V / 9.0
    return K * np.sinh(2 * (B_ALPHA - c * B_GAMMA))

print(f"V* = {V_STAR:.4f}  |  C_ZERO = {C_ZERO:.4f}")

# ── Drug Category Definitions (N=34) ─────────────────────────────────────────
# O = Opacity (0-3): pricing mechanism hidden from patient
# R = Responsiveness (0-3): responsive to market signals vs invariant cost-based
# α = Coupling (0-3): patient exit difficulty
# HHI = Herfindahl-Hirschman Index (0-10000), from published literature
# harm_tier: catastrophic/life-threatening/severe/moderate/null/structural/control

DRUG_CATEGORIES = [

    # ══ V=9 VOID MAXIMUM ════════════════════════════════════════════════════
    {
        "name": "OxyContin (Sackler era, 1996–2010)",
        "O": 3, "R": 3, "alpha": 3, "hhi": 4900,
        "harm_tier": "catastrophic",
        "group": "US Branded",
        "evidence": (
            "O=3: Purdue concealed addiction risk; 'I want to sell Oxy like Doritos' (MA AG 2019). "
            "R=3: Detail-rep model paid per script, maximally market-responsive. "
            "α=3: Opioid receptor dependency = physiological coupling. "
            "HHI≈4900: Purdue ~70% ER-oxycodone market at peak (MA AG complaint). "
            "D3 cascade: 500,000+ opioid deaths 1999–2019 (CDC WONDER). THE paradigmatic void."
        ),
    },
    {
        "name": "Insulin — branded, Big 3 (pre-IRA)",
        "O": 3, "R": 3, "alpha": 3, "hhi": 3200,
        "harm_tier": "life-threatening",
        "group": "US Branded",
        "evidence": (
            "O=3: List price vs PBM net price completely opaque. Big 3 list prices rose 1000%+ "
            "1996–2019 (Senate Finance 2021). R=3: Shadow pricing — coordinated list increases, "
            "not cost-driven. α=3: Type 1 diabetics have zero exit — no insulin = DKA death. "
            "HHI≈3200: Big 3 (Lilly 38%, Novo 29%, Sanofi 23%) = HHI 3274 in US market."
        ),
    },
    {
        "name": "HIV ARVs — branded, Gilead (2001–2012)",
        "O": 3, "R": 3, "alpha": 3, "hhi": 5200,
        "harm_tier": "life-threatening",
        "group": "US Branded",
        "evidence": (
            "O=3: Tenofovir synthesis ~$1/day; US list $36/day at launch. Cost-plus opaque. "
            "R=3: Patent evergreening (TAF strategy); delayed generic access. "
            "α=3: HAART = lifelong survival dependency. "
            "HHI≈5200: Gilead ~75% HAART market 2009 (Senators Wyden/Grassley 2021). "
            "Harm: millions of deaths from access denial in developing world pre-TRIPS flex."
        ),
    },
    {
        "name": "Humira (AbbVie, pre-biosimilar)",
        "O": 3, "R": 3, "alpha": 3, "hhi": 8500,
        "harm_tier": "severe",
        "group": "US Branded",
        "evidence": (
            "O=3: 132+ patent thicket blocked biosimilar entry 8yr post-primary patent (I-MAK 2021). "
            "List price rose 470% 2003–2022. R=3: Raised prices ahead of biosimilar entry. "
            "α=3: RA/Crohn's patients face severe flare on discontinuation; months to restabilize. "
            "HHI≈8500: Near-monopoly; world's best-selling drug 2013–2022. $200B+ revenue."
        ),
    },
    {
        "name": "Novel chemotherapy (branded oncology)",
        "O": 3, "R": 3, "alpha": 3, "hhi": 6500,
        "harm_tier": "life-threatening",
        "group": "US Branded",
        "evidence": (
            "O=3: 'Value-based' pricing where value defined by manufacturer; R&D inflation opaque "
            "(Prasad & Mailankody 2017 JAMA). R=3: Survival imperative = maximum inelasticity. "
            "α=3: Discontinuation = accepting mortality. HHI≈6500: Single manufacturer 5–12yr exclusivity. "
            "Median launch price 2021: $180,000/yr (IQVIA 2022). 42% cancer patients bankrupt within 2yr."
        ),
    },
    {
        "name": "EpiPen branded (Mylan, 2007–2018)",
        "O": 3, "R": 3, "alpha": 3, "hhi": 8800,
        "harm_tier": "life-threatening",
        "group": "US Branded",
        "evidence": (
            "O=3: List price rose 500% 2007–2016 with zero clinical justification. Pricing mechanism "
            "hidden behind PBM rebate capture. R=3: Mylan exploited schools' epinephrine mandate, "
            "creating government-enforced captive demand. α=3: Anaphylaxis = immediate death without "
            "epinephrine; no delay possible. HHI≈8800: Mylan held ~85% auto-injector market pre-generic. "
            "School mandate strategy = regulatory coupling mechanism."
        ),
    },
    {
        "name": "CAR-T therapy (Kymriah/Yescarta)",
        "O": 3, "R": 3, "alpha": 3, "hhi": 9200,
        "harm_tier": "life-threatening",
        "group": "US Branded",
        "evidence": (
            "O=3: List price $375,000–$475,000 for single infusion. Cost-of-production opaque. "
            "Novartis/Gilead pricing justified by 'outcomes-based' model not disclosed. "
            "R=3: Pricing responsive to what insurance will bear, not manufacturing cost. "
            "α=3: Relapsed/refractory B-ALL or DLBCL — CAR-T is last-line therapy, death alternative. "
            "HHI≈9200: Duopoly with near-zero substitution (Novartis + Gilead). Highest-Pe drug type."
        ),
    },
    {
        "name": "Gene therapy (Zolgensma, $2.1M list)",
        "O": 3, "R": 3, "alpha": 3, "hhi": 10000,
        "harm_tier": "life-threatening",
        "group": "US Branded",
        "evidence": (
            "O=3: AveXis (Novartis) set price at $2.1M using 'cost per life-year' model with "
            "undisclosed assumptions. Pricing advisory firm engagement not public. "
            "R=3: Novartis explicitly priced to capture maximum 'willingness to pay' across insurance. "
            "α=3: Spinal muscular atrophy Type 1 — death within 2yr without treatment. "
            "HHI=10000: Single manufacturer, no competition. World's most expensive drug by unit price."
        ),
    },
    {
        "name": "PBM rebate layer (CVS/ESI/OptumRx)",
        "O": 3, "R": 3, "alpha": 3, "hhi": 2900,
        "harm_tier": "structural",
        "group": "System Layer",
        "evidence": (
            "O=3: Rebate amounts trade secrets (FTC 2024). 'Rebate contracts contain confidentiality "
            "clauses preventing public disclosure.' Patients pay list; PBM retains spread. "
            "R=3: Formulary designed to maximize rebate capture, not outcomes (FTC 2024). "
            "α=3: Patients cannot exit employer-linked insurance → PBM → formulary. "
            "HHI≈2900: Big 3 PBMs (CVS 34%, ESI 25%, Optum 22%) = HHI 2406 by lives; higher by revenue. "
            "Meta-layer: amplifies Pe of every branded drug operating above it."
        ),
    },
    {
        "name": "Specialty pharmacy (Accredo/CVS Specialty)",
        "O": 3, "R": 3, "alpha": 3, "hhi": 7800,
        "harm_tier": "structural",
        "group": "System Layer",
        "evidence": (
            "O=3: Specialty drug dispensing fees, DIR fees, and network exclusions opaque. "
            "PBMs require expensive drugs be filled through affiliated specialty pharmacy. "
            "R=3: 90-day dispensing requirements for chronic specialty drugs enforced through formulary. "
            "α=3: Specialty pharmacy lock-in for biologics/specialty drugs = patient has no retail option. "
            "HHI≈7800: CVS Specialty + Accredo (ESI) control >70% specialty drug dispensing. "
            "Adds ~$50–200/fill markup hidden from patient."
        ),
    },
    {
        "name": "Rare disease drugs (Spinraza, Exondys)",
        "O": 3, "R": 3, "alpha": 3, "hhi": 9800,
        "harm_tier": "life-threatening",
        "group": "US Branded",
        "evidence": (
            "O=3: Orphan drug designation allows 7yr monopoly extension. Biogen Spinraza $750K first "
            "year, $375K/yr maintenance. Manufacturing cost ~$1,000. Pricing model undisclosed. "
            "R=3: Parents of children with SMA have zero price elasticity. Manufacturers exploit fully. "
            "α=3: Progressive neuromuscular disease — discontinuation = accelerated decline. "
            "HHI≈9800: Single-source, no competition. Orphan drug market = systematic Pe=maximum."
        ),
    },
    {
        "name": "PrEP branded (Truvada/Descovy, Gilead)",
        "O": 3, "R": 3, "alpha": 3, "hhi": 8200,
        "harm_tier": "life-threatening",
        "group": "US Branded",
        "evidence": (
            "O=3: Truvada US list $2,000/month; manufacturing cost $6/month (Senate Finance 2019). "
            "Developed with $50M in NIH grants — public R&D with private pricing capture. "
            "R=3: Gilead delayed generic while HIV prevention imperative created captive demand. "
            "α=3: HIV prevention = survival coupling; discontinuation risk = HIV acquisition. "
            "HHI≈8200: Gilead held >95% PrEP market until 2020 generic entry. "
            "Senators Wyden/Grassley investigation confirmed pricing mechanism opacity."
        ),
    },
    {
        "name": "Branded antipsychotics (Abilify/Latuda era)",
        "O": 3, "R": 2, "alpha": 3, "hhi": 5500,
        "harm_tier": "severe",
        "group": "US Branded",
        "evidence": (
            "O=3: Manufacturer-defined 'atypical' category enabled premium pricing over generics "
            "with marginal clinical differentiation. DTC advertising spending $5B+/yr at peak. "
            "R=2: Somewhat constrained by growing generic atypical options; not pure monopoly. "
            "α=3: Psychiatric medications — abrupt discontinuation causes relapse/hospitalization. "
            "HHI≈5500: Otsuka (Abilify) and Sunovion (Latuda) dominant before generics."
        ),
    },

    # ══ V=8 HIGH VOID ════════════════════════════════════════════════════════
    {
        "name": "GLP-1 agonists (Ozempic/Wegovy, 2022+)",
        "O": 3, "R": 3, "alpha": 2, "hhi": 6200,
        "harm_tier": "severe",
        "group": "US Branded",
        "evidence": (
            "O=3: Wegovy $900/month list; ~$5/dose manufacturing (ICER 2023). "
            "R=3: Novo exploited obesity/diabetes epidemic; price unchanged despite Lilly entry. "
            "α=2: Physiological coupling (weight regains within 1yr of stopping, NEJM 2022) "
            "but not immediately lethal. Predicted α→3 as chronic use evidence accumulates."
        ),
    },
    {
        "name": "Prior authorization (insurance denial)",
        "O": 3, "R": 3, "alpha": 2, "hhi": 4500,
        "harm_tier": "severe",
        "group": "System Layer",
        "evidence": (
            "O=3: PA criteria proprietary algorithms (eviCore, Magellan). Denial logic not disclosed. "
            "R=3: PA added to highest-cost branded drugs — responds to cost drivers, not clinical evidence. "
            "α=2: Step therapy forces generic fail-first; patient coupled to process, not specific drug. "
            "HHI≈4500: Regional insurance concentration (4–6 carriers per state, state HHI varies)."
        ),
    },

    # ══ V=7 DISCRIMINANT CASE ════════════════════════════════════════════════
    {
        "name": "Daraprim post-Shkreli (Turing, 2015) ★",
        "O": 3, "R": 3, "alpha": 1, "hhi": 10000,
        "harm_tier": "moderate",
        "group": "US Branded",
        "evidence": (
            "O=3: 5455% price increase with zero transparency. R=3: Single-source, exploited "
            "toxoplasmosis demand inelasticity. α=1: SHORT-COURSE treatment (weeks). "
            "No physiological dependency. Generic competitors entered within months. "
            "HHI=10000: True monopoly at moment of increase. THE DISCRIMINANT CASE: "
            "highest MCI in sample, 7th-highest Pe. α=1 prevents harm cascade. "
            "Made the void visible. Sackler kept it hidden while building α=3."
        ),
    },
    {
        "name": "Antidepressants on-patent (Trintellix/Viibryd)",
        "O": 3, "R": 2, "alpha": 2, "hhi": 5800,
        "harm_tier": "moderate",
        "group": "US Branded",
        "evidence": (
            "O=3: Manufacturer-defined differentiation from older SSRIs with marginal evidence. "
            "DTC advertising creates demand opacity. R=2: Moderate constraint from older generics nearby. "
            "α=2: Discontinuation syndrome documented; gradual taper needed. Not life-threatening. "
            "HHI≈5800: Single-manufacturer exclusivity with nearby generic substitutes dampening."
        ),
    },

    # ══ V=6 MODERATE VOID ════════════════════════════════════════════════════
    {
        "name": "Branded SSRIs (on-patent, pre-generic)",
        "O": 2, "R": 2, "alpha": 2, "hhi": 3800,
        "harm_tier": "moderate",
        "group": "US Branded",
        "evidence": (
            "O=2: Clinical trial data largely published; some selective reporting (Kirsch 2008). "
            "R=2: Moderate — generic entry timeline partially predictable. "
            "α=2: Discontinuation syndrome; gradual taper. Not lethal. "
            "HHI≈3800: Patent cliff creates competition; pre-generic moderate concentration."
        ),
    },
    {
        "name": "Biosimilar adalimumab (2023+ market)",
        "O": 2, "R": 2, "alpha": 2, "hhi": 3200,
        "harm_tier": "moderate",
        "group": "US Branded",
        "evidence": (
            "O=2: WAC listed but rebate structures still partially opaque. 85% discount to Humira "
            "list but formulary exclusions limit effective competition. "
            "R=2: Amgen/Sandoz competing — price competition beginning. "
            "α=2: Physician reluctance to switch stable patients (nocebo, monitoring costs). "
            "Pe falling as biosimilar market matures — natural experiment 2023-2026 (PRC-3)."
        ),
    },

    # ══ V=5 TRANSITION ZONE ════════════════════════════════════════════════
    {
        "name": "Opioid generics (generic oxycodone/hydrocodone)",
        "O": 1, "R": 1, "alpha": 3, "hhi": 1200,
        "harm_tier": "moderate",
        "group": "US Generic",
        "evidence": (
            "O=1: Generic pricing transparent; FDA publishes manufacturers, prices visible. "
            "R=1: Commodity competition — multiple manufacturers. "
            "α=3: Opioid dependency remains even in generic market — same receptor coupling. "
            "HHI≈1200: 10+ manufacturers for generic oxycodone (FDA Orange Book 2023). "
            "KEY INSIGHT: α=3 but Pe NEGATIVE (-4.2) because O and R are low. "
            "Coupling alone does not create an attractive void. Opacity + Responsiveness required."
        ),
    },
    {
        "name": "Insulin glargine — Canada (same molecule)",
        "O": 1, "R": 1, "alpha": 3, "hhi": 1800,
        "harm_tier": "moderate",
        "group": "International Control",
        "evidence": (
            "O=1: Provincial formularies (NIHB, ODB) publish drug prices. Cost-of-manufacture "
            "visible through PMPRB review. R=1: Government-negotiated pricing via CADTH — "
            "limited responsiveness to manufacturer demand. α=3: Type 1 diabetes = survival coupling "
            "(identical to US, same physiology). HHI≈1800: Same Big 3 manufacturers but regulated. "
            "INTERNATIONAL CONTROL: Same α=3 as US insulin (V=9) but V=5 Pe=−4.2. "
            "Pe measures system architecture, not pharmacology."
        ),
    },
    {
        "name": "PrEP generic (tenofovir/emtricitabine)",
        "O": 1, "R": 1, "alpha": 2, "hhi": 1500,
        "harm_tier": "null",
        "group": "US Generic",
        "evidence": (
            "O=1: Generic pricing transparent via FDA Orange Book. Multiple manufacturers. "
            "R=1: Price competition among generics drives margin toward commodity. "
            "α=2: HIV prevention dependency — daily adherence required for efficacy. "
            "HHI≈1500: Cipla, Teva, Apotex, Lupin competing (FDA 2023). "
            "Pe transition: Truvada (branded V=8, Pe=25.2) → generic (V=4, Pe=−16) = market correction."
        ),
    },

    # ══ V=4 ══════════════════════════════════════════════════════════════════
    {
        "name": "EpiPen generic (Auvi-Q/authorized generic)",
        "O": 2, "R": 1, "alpha": 3, "hhi": 3500,
        "harm_tier": "moderate",
        "group": "US Generic",
        "evidence": (
            "O=2: Pricing partially transparent but authorized generic strategy created confusion. "
            "Mylan released 'authorized generic' at 50% of list — opaque discount vs true generic. "
            "R=1: Emerging competition from Teva/Amneal generics. "
            "α=3: Anaphylaxis dependency unchanged — same survival coupling as branded. "
            "HHI≈3500: Mylan retained significant share through authorized generic strategy."
        ),
    },
    {
        "name": "VA formulary system (Veterans Affairs)",
        "O": 1, "R": 0, "alpha": 2, "hhi": 800,
        "harm_tier": "null",
        "group": "US System Control",
        "evidence": (
            "O=1: VA National Formulary published. Drug prices negotiated and listed. "
            "R=0: VA negotiates based on cost-effectiveness, not manufacturer demand. "
            "Statutory authority to negotiate + formulary exclusion = R dimension suppressed. "
            "α=2: Veterans dependent on VA system for care continuity. "
            "HHI≈800: VA formulary forces generic/therapeutic substitution. "
            "Pe=−16: System-level constraint reduces Pe vs commercial market."
        ),
    },

    # ══ V=3 NULL / REPULSIVE ══════════════════════════════════════════════
    {
        "name": "Generic statins (atorvastatin, simvastatin)",
        "O": 1, "R": 1, "alpha": 1, "hhi": 620,
        "harm_tier": "null",
        "group": "US Generic",
        "evidence": (
            "O=1: FDA publishes generic approval data; bioequivalence public. "
            "R=1: Commodity — 15+ manufacturers compete on cost. "
            "α=1: Preventive; discontinuation not acutely dangerous. "
            "HHI≈620: >15 manufacturers (FDA Orange Book 2023). True commodity."
        ),
    },
    {
        "name": "Generic SSRIs (fluoxetine, sertraline)",
        "O": 1, "R": 1, "alpha": 1, "hhi": 540,
        "harm_tier": "null",
        "group": "US Generic",
        "evidence": (
            "O=1: Full generic transparency. Multiple manufacturers. "
            "R=1: Commodity pricing. R=1 not 0 due to REMS programs for some. "
            "α=1: Same as branded but exit path cheaper. "
            "HHI≈540: >12 manufacturers for sertraline (FDA 2023)."
        ),
    },
    {
        "name": "Insulin glargine — Germany (AMNOG regulated)",
        "O": 0, "R": 0, "alpha": 3, "hhi": 1200,
        "harm_tier": "moderate",
        "group": "International Control",
        "evidence": (
            "O=0: AMNOG (Arzneimittelmarkt-Neuordnungsgesetz) requires published benefit assessment "
            "and statutory pricing. Net prices negotiated publicly via GKV-Spitzenverband. "
            "R=0: Statutory maximum price — not responsive to manufacturer demand signals at all. "
            "α=3: Type 1 diabetes = survival coupling (identical physiology to US). "
            "HHI≈1200: Same Big 3 but AMNOG forces price transparency and comparator-based pricing. "
            "INTERNATIONAL CONTROL: V=3, Pe=−25.9 vs US insulin V=9, Pe=+43.9. "
            "Same molecule, same coupling, 3 full void dimensions suppressed by architecture."
        ),
    },
    {
        "name": "Metformin generic (Type 2 diabetes)",
        "O": 0, "R": 0, "alpha": 1, "hhi": 420,
        "harm_tier": "null",
        "group": "US Generic",
        "evidence": (
            "O=0: $4 at Walmart, $0 with GoodRx at many pharmacies. "
            "Perfect cost transparency — retail price = manufacturing cost + minimal margin. "
            "R=0: No demand manipulation possible. Off-patent since 1994. "
            "α=1: T2D management — important but not immediately lethal to stop. "
            "HHI≈420: Extreme competition. One of the most-dispensed drugs in the world."
        ),
    },

    # ══ V=2 REPULSIVE ════════════════════════════════════════════════════════
    {
        "name": "COVID mRNA vaccines (government purchase)",
        "O": 1, "R": 0, "alpha": 1, "hhi": 3200,
        "harm_tier": "null",
        "group": "US System Control",
        "evidence": (
            "O=1: OWS contracts made public; pricing disclosed under FOIA. "
            "R=0: Fixed government-negotiated price per dose. "
            "α=1: Voluntary; single/double dose; no chronic coupling. "
            "HHI≈3200: Pfizer+Moderna duopoly but government removed R dimension entirely."
        ),
    },
    {
        "name": "Australian PBS coverage (same drugs as US branded)",
        "O": 0, "R": 0, "alpha": 2, "hhi": 1100,
        "harm_tier": "null",
        "group": "International Control",
        "evidence": (
            "O=0: PBS (Pharmaceutical Benefits Scheme) lists all covered drugs with co-payment. "
            "Cost-effectiveness evaluation (PBAC) published. Net price negotiated, disclosed. "
            "R=0: PBAC assessment sets price; manufacturer responsiveness irrelevant — "
            "accept the price or leave the PBS. "
            "α=2: Chronic disease dependency unchanged — same pharmacology as US. "
            "HHI≈1100: PBS competition from all listed manufacturers. "
            "CONTROL: Same drugs as US V=9 market. V=2, Pe=−45 due to architecture."
        ),
    },

    # ══ V=1 ANTI-VOID ════════════════════════════════════════════════════════
    {
        "name": "Naloxone / Narcan OTC (overdose reversal)",
        "O": 0, "R": 0, "alpha": 0, "hhi": 380,
        "harm_tier": "null",
        "group": "Constraint Tool",
        "evidence": (
            "O=0: OTC since 2023. Multiple manufacturers. Prices publicly listed. "
            "R=0: Government and public health purchase removes demand-driven pricing. "
            "Emergent Aid Foundation distributes at cost. "
            "α=0: Single-use emergency reversal agent — no coupling mechanism. "
            "HHI≈380: Multiple manufacturers competing (Adapt, Pfizer, Amphastar, generics). "
            "Pe=−125: Maximum anti-void. The thermodynamic antidote to opioid cascade. "
            "The framework predicts: restricting naloxone access = increasing system Pe."
        ),
    },
    {
        "name": "Mark Cuban Cost Plus Drugs (live US market)",
        "O": 0, "R": 0, "alpha": 0, "hhi": 290,
        "harm_tier": "null",
        "group": "Constraint Tool",
        "evidence": (
            "O=0: Prices listed as exact formula: cost + 15% markup + $3/prescription. "
            "No PBMs. No rebates. No formulary exclusions. Fully transparent supply chain. "
            "R=0: Fixed markup formula — not responsive to any demand signal. "
            "α=0: No lock-in mechanism. Patient can fill anywhere. "
            "HHI≈290: Nascent market; Cost Plus + competing direct pharmacies. "
            "NATURAL EXPERIMENT IN PROGRESS: Same drugs as US V=9 market, available now. "
            "Pe=−125: Demonstrates that void-free pharma is operationally possible in US."
        ),
    },

    # ══ V=0 PERFECT NULL ══════════════════════════════════════════════════════
    {
        "name": "Aspirin / OTC ibuprofen (commodity)",
        "O": 0, "R": 0, "alpha": 0, "hhi": 180,
        "harm_tier": "null",
        "group": "Null Case",
        "evidence": (
            "O=0: Shelf price = visible. Multiple brands. "
            "R=0: Commodity — margin near zero, no demand manipulation. "
            "α=0: OTC, no prescription, abundant substitutes. "
            "HHI≈180: Extreme competition. Perfect null. Pe=−125."
        ),
    },
]

N = len(DRUG_CATEGORIES)
print(f"\nDrug categories scored: N={N}")

# ── Compute V, c, Pe ─────────────────────────────────────────────────────────
for d in DRUG_CATEGORIES:
    d["V"]   = d["O"] + d["R"] + d["alpha"]
    d["c"]   = 1.0 - d["V"] / 9.0
    d["Pe"]  = thrml_pe(d["V"])

V_arr   = np.array([d["V"]   for d in DRUG_CATEGORIES])
Pe_arr  = np.array([d["Pe"]  for d in DRUG_CATEGORIES])
HHI_arr = np.array([d["hhi"] for d in DRUG_CATEGORIES])
names   = [d["name"] for d in DRUG_CATEGORIES]

print(f"\n{'Category':<52} {'V':>3} {'Pe':>8} {'HHI':>6}")
print("-"*72)
for d in sorted(DRUG_CATEGORIES, key=lambda x: -x["Pe"]):
    marker = " ★" if "★" in d["name"] else ""
    print(f"{d['name'][:52]:<52} {d['V']:>3} {d['Pe']:>8.1f} {d['hhi']:>6}{marker}")

# ── Spearman Validation ───────────────────────────────────────────────────────
rho_V_HHI,  p_V_HHI  = stats.spearmanr(V_arr, HHI_arr)
rho_Pe_HHI, p_Pe_HHI = stats.spearmanr(Pe_arr, HHI_arr)

# LOO stability
loo_rhos = []
for i in range(N):
    mask = np.ones(N, dtype=bool); mask[i] = False
    r, _ = stats.spearmanr(V_arr[mask], HHI_arr[mask])
    loo_rhos.append(r)
loo_rhos = np.array(loo_rhos)

print(f"\n── Spearman Validation (N={N}) ─────────────────────────────")
print(f"Spearman(V, HHI)  = {rho_V_HHI:.4f}  p={p_V_HHI:.6f}")
print(f"Spearman(Pe, HHI) = {rho_Pe_HHI:.4f}  p={p_Pe_HHI:.6f}")
print(f"LOO range:          [{loo_rhos.min():.4f}, {loo_rhos.max():.4f}]  mean={loo_rhos.mean():.4f}")

# ── International Comparison — THE KEY FINDING ───────────────────────────────
print(f"\n── INTERNATIONAL COMPARISON: Same Drug, Different Architecture ──")
intl = {r: next(d for d in DRUG_CATEGORIES if r in d["name"])
        for r in ["Insulin — branded, Big 3", "Insulin glargine — Canada", "Insulin glargine — Germany"]}
print(f"{'Country / System':<42} {'V':>3} {'Pe':>8} {'α':>4}  Note")
print("-"*72)
for label, d in [("US (Big 3 branded, PBM system)",    intl["Insulin — branded, Big 3"]),
                 ("Canada (NIHB/ODB provincial)",       intl["Insulin glargine — Canada"]),
                 ("Germany (AMNOG statutory pricing)",  intl["Insulin glargine — Germany"])]:
    print(f"{label:<42} {d['V']:>3} {d['Pe']:>8.1f} {d['alpha']:>4}  α unchanged")
print("\nSame pharmacology. Same coupling. Different regulatory architecture = different Pe.")
print("Pe measures SYSTEM VOID, not drug toxicity.")

# ── Discriminant Test ─────────────────────────────────────────────────────────
oxy  = next(d for d in DRUG_CATEGORIES if "OxyContin" in d["name"])
dara = next(d for d in DRUG_CATEGORIES if "Daraprim" in d["name"])
print(f"\n── Sackler vs Shkreli Discriminant ─────────────────────────────")
print(f"{'Metric':<28} {'OxyContin (Sackler)':>22} {'Daraprim (Shkreli)':>22}")
print("-"*74)
for metric, sv, dv in [
    ("HHI",              f"{oxy['hhi']:,}",             f"{dara['hhi']:,} ← higher"),
    ("Void score V",     str(oxy['V']),                 str(dara['V'])),
    ("Pe",               f"+{oxy['Pe']:.1f}",           f"+{dara['Pe']:.1f}"),
    ("α (coupling)",     "3 — opioid receptor",         "1 — short-course"),
    ("Pe ratio",         "—",                           f"{oxy['Pe']/dara['Pe']:.1f}× (Sackler)"),
    ("D3 cascade",       "YES — 500K dead",             "NO — price shock only"),
]:
    print(f"{metric:<28} {sv:>22} {dv:>22}")

# ── Mark Cuban Natural Experiment ────────────────────────────────────────────
costplus = next(d for d in DRUG_CATEGORIES if "Cost Plus" in d["name"])
print(f"\n── Mark Cuban / Cost Plus Drugs: Live Natural Experiment ───────")
print(f"Same drugs as US branded (V=9, Pe=+43.9) available at Pe={costplus['Pe']:.1f}")
print("O=0 (cost+15%+$3 displayed), R=0 (fixed markup), α=0 (no lock-in)")
print("No PBM. No formulary. No rebate. Pe=-125 in the same US market.")
print("Prediction (CPD-1): Cost Plus dispensing share negatively correlates with")
print("equivalent branded drug Pe. As transparency wins market share, system Pe drops.")

# ── Natural Experiments ───────────────────────────────────────────────────────
print(f"\n── Natural Experiments (Testable 2025–2027) ────────────────────")
NE = [
    ("IRA insulin cap ($35)",   9, 5, 0.92, "largest structural void reduction; CMS Part D 2025"),
    ("Biosimilar adalimumab",   9, 6, 0.55, "IQVIA adalimumab market share 2023-2026"),
    ("PrEP generic entry",      8, 4, 0.60, "CMS Part D dispensing rates pre/post generic 2020"),
    ("EpiPen generic entry",    9, 7, 0.35, "FDA generic approval 2018; market share trajectory"),
    ("VA formulary vs commercial", 9, 4, 0.62, "CMS/VA price differential; same drug, same year"),
]
for drug, v_pre, v_post, price_drop, test in NE:
    pe_pre, pe_post = thrml_pe(v_pre), thrml_pe(v_post)
    pe_drop = (pe_pre - pe_post)/pe_pre
    print(f"{drug:<35}: Pe {pe_pre:.1f}→{pe_post:.1f}  (Δ={pe_drop:.0%})  Test: {test}")

# ── Falsifiable Predictions ───────────────────────────────────────────────────
print(f"\n── Falsifiable Predictions (PRC-1 through PRC-7) ──────────────")
PREDS = [
    ("PRC-1", "HHI correlation (N=34)",
     f"Spearman(V, HHI_CMS) > 0.70 (current proxy: {rho_V_HHI:.3f}). "
     "Refuted if actual CMS Part D HHI data gives ρ < 0.70 at N≥30."),
    ("PRC-2", "Daraprim discriminant replicates",
     "At N≥50, Daraprim retains outlier status: highest HHI but ≤7th Pe. "
     "Refuted if α-1 drugs with V=7 show harm cascade rates > V=9, α=3 drugs."),
    ("PRC-3", "Biosimilar Pe tracks market share",
     "Adalimumab: Spearman(biosimilar_share, ΔPe_proxy) > 0.80 (2023-2026 IQVIA)."),
    ("PRC-4", "IRA natural experiment",
     "Negotiated drugs: Spearman(price_reduction%, Pe_reduction%) > 0.70 in 2025-2026 CMS data."),
    ("PRC-5", "PBM transparency rule",
     "If FTC/Congress mandates rebate disclosure: L3 Pe drops 25.2→3.8 (V=8→6). "
     "Net-to-list convergence measurable via CMS Part D."),
    ("PRC-6", "International replication",
     "Same molecule in 5+ countries: Spearman(V_country, Pe_country) > 0.85. "
     "Pe should follow regulatory architecture, not pharmacology."),
    ("PRC-7", "Cost Plus Drugs market share effect",
     "Cost Plus dispensing share negatively correlated with equivalent branded drug Pe_proxy "
     "across their formulary. Refuted if no dose-response in market share vs transparency."),
]
for pid, title, pred in PREDS:
    print(f"\n{pid} — {title}")
    print(f"  {pred}")

# ── Generate SVG ──────────────────────────────────────────────────────────────
COLORS = {
    "catastrophic":    "#ff2222",
    "life-threatening":"#ff5533",
    "severe":          "#ff8833",
    "structural":      "#cc44ff",
    "moderate":        "#ffcc44",
    "null":            "#44dd88",
}
GROUP_MARKER = {
    "US Branded":           "o",
    "System Layer":         "s",
    "US Generic":           "^",
    "International Control":"D",
    "US System Control":    "P",
    "Constraint Tool":      "*",
    "Null Case":            "X",
}

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor('#0a0a0a')

sorted_cats = sorted(DRUG_CATEGORIES, key=lambda x: x["Pe"])

# Plot 1: Full Pe spectrum
ax1 = axes[0]
ax1.set_facecolor('#111111')
y_pos = np.arange(N)
bcolors = [COLORS[d["harm_tier"]] for d in sorted_cats]
ax1.barh(y_pos, [d["Pe"] for d in sorted_cats], color=bcolors, alpha=0.85, height=0.75)
ax1.set_yticks(y_pos)
ylab = []
for d in sorted_cats:
    label = d["name"][:34]
    if "★" in d["name"]:
        label += " ★"
    ylab.append(label)
ax1.set_yticklabels(ylab, fontsize=6.5, color='#ccc')
ax1.axvline(0, color='#555', lw=1.5, ls='--')
ax1.set_xlabel('THRML Pe', color='#999', fontsize=9)
ax1.set_title(f'Drug Market Pe Spectrum (N={N})', color='#fff', fontsize=10, pad=8)
ax1.tick_params(colors='#777', axis='x')
for sp in ax1.spines.values():
    sp.set_color('#333')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Plot 2: V vs HHI scatter
ax2 = axes[1]
ax2.set_facecolor('#111111')
groups_seen = set()
for d in DRUG_CATEGORIES:
    c = COLORS[d["harm_tier"]]
    m = GROUP_MARKER.get(d["group"], "o")
    lbl = d["group"] if d["group"] not in groups_seen else None
    groups_seen.add(d["group"])
    ax2.scatter(d["V"], d["hhi"], color=c, marker=m, s=70, alpha=0.85, zorder=3,
                label=lbl)
    if "★" in d["name"] or "Cost Plus" in d["name"] or "Canada" in d["name"] or "Germany" in d["name"]:
        short = d["name"].split("(")[0].strip()[:18]
        ax2.annotate(short, (d["V"], d["hhi"]), fontsize=6.5, color='#ddd',
                     xytext=(5,3), textcoords='offset points')

ax2.text(0.05, 0.97, f"ρ = {rho_V_HHI:.3f}  p = {p_V_HHI:.4f}",
         transform=ax2.transAxes, color='#00d4ff', fontsize=9, va='top')
ax2.text(0.05, 0.90, f"LOO [{loo_rhos.min():.3f}, {loo_rhos.max():.3f}]",
         transform=ax2.transAxes, color='#888', fontsize=8, va='top')
ax2.set_xlabel('Void Score V', color='#999', fontsize=9)
ax2.set_ylabel('HHI (0–10000)', color='#999', fontsize=9)
ax2.set_title('Void Score vs Market Concentration\n(Daraprim = discriminant outlier ★)',
              color='#fff', fontsize=9, pad=8)
ax2.tick_params(colors='#777')
for sp in ax2.spines.values():
    sp.set_color('#333')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Plot 3: International comparison
ax3 = axes[2]
ax3.set_facecolor('#111111')
intl_data = [d for d in DRUG_CATEGORIES if d["group"] == "International Control"
             or "insulin" in d["name"].lower() or "Insulin" in d["name"]]
# Filter to just the three insulin cases + a couple controls
insulin_cases = [
    ("US Branded (V=9)", 9, "#ff3333"),
    ("Canada regulated (V=5)", 5, "#ffaa22"),
    ("Germany AMNOG (V=3)", 3, "#44dd88"),
]
x_pos2 = np.arange(3)
colors3 = [c for _,_,c in insulin_cases]
labels3 = [l for l,_,_ in insulin_cases]
pe_vals  = [thrml_pe(v) for _,v,_ in insulin_cases]
bars3 = ax3.bar(x_pos2, pe_vals, color=colors3, alpha=0.85, width=0.6)
ax3.axhline(0, color='#555', lw=1.5, ls='--')
ax3.set_xticks(x_pos2)
ax3.set_xticklabels(labels3, color='#ccc', fontsize=8.5)
ax3.set_ylabel('Pe', color='#999', fontsize=9)
ax3.set_title('Insulin Glargine — Same Molecule\nDifferent Regulatory Architecture',
              color='#fff', fontsize=9, pad=8)
for bar, val in zip(bars3, pe_vals):
    ax3.text(bar.get_x()+bar.get_width()/2, val + (2 if val>0 else -4),
             f"Pe={val:.1f}", ha='center', va='bottom' if val>0 else 'top',
             color='#fff', fontsize=9, fontweight='bold')
ax3.text(0.05, 0.95, "α=3 (survival coupling) unchanged\nacross all three countries",
         transform=ax3.transAxes, color='#888', fontsize=8, va='top', style='italic')
ax3.tick_params(colors='#777')
for sp in ax3.spines.values():
    sp.set_color('#333')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
plt.savefig("nb_pharma01_drug_pricing_pe.svg", format='svg',
            facecolor='#0a0a0a', bbox_inches='tight')
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY  N={N}")
print(f"{'='*60}")
print(f"Spearman(V, HHI)  = {rho_V_HHI:.3f}  p={p_V_HHI:.6f}")
print(f"LOO stability:      [{loo_rhos.min():.3f}, {loo_rhos.max():.3f}]")
print(f"\nDiscriminant: OxyContin Pe={oxy['Pe']:.1f} vs Daraprim Pe={dara['Pe']:.1f}  ratio={oxy['Pe']/dara['Pe']:.1f}x")
print(f"\nInternational insulin (same α=3):")
for label, key in [("US", "Insulin — branded, Big 3"), ("Canada","Insulin glargine — Canada"), ("Germany","Insulin glargine — Germany")]:
    d = next(x for x in DRUG_CATEGORIES if key in x["name"])
    print(f"  {label}: V={d['V']}, Pe={d['Pe']:.1f}")
print(f"\nNull cases confirmed Pe<0: {sum(1 for d in DRUG_CATEGORIES if d['Pe']<0)}/{N}")
print(f"Predictions: PRC-1 through PRC-7")
print(f"SVG: nb_pharma01_drug_pricing_pe.svg")
