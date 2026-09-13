"""
Real MPLADS reference data.

Categories are taken verbatim from Annexure-VIII of the MPLADS
Guidelines 2023 (the official Indicative List of permissible works),
including their PFMS codes and Annexure clause numbers.

Nothing here is invented except the unit rates, which are plausible
market estimates used only to generate synthetic amounts.
"""

# ─────────────────────────────────────────────────────────────
# ANNEXURE-VIII WORKS  (subset with measurable units)
#
# key = PFMS code
# (parent activity, activity name, annexure ref, unit, rate, qty range)
# ─────────────────────────────────────────────────────────────
WORKS = {
    "5.001": ("Drinking water and sanitation",
              "Installing tube-wells and borewells",
              "5.1", "borewell", 320_000, (1, 6)),

    "5.002": ("Drinking water and sanitation",
              "Installing hand pumps",
              "5.2", "hand pump", 85_000, (2, 12)),

    "5.003": ("Drinking water and sanitation",
              "Construction of water tanks",
              "5.3", "kilolitre", 22_000, (10, 80)),

    "1.5.7": ("Drinking water and sanitation",
              "Providing drains and gutters for public drainage",
              "5.7", "metre", 3_400, (150, 1800)),

    "1.5.8": ("Drinking water and sanitation",
              "Construction of public toilets and bathrooms",
              "5.8", "seat", 65_000, (4, 30)),

    "3.001": ("Education",
              "Construction of rooms and halls in school and colleges",
              "3.1", "room", 850_000, (1, 5)),

    "1.3.15": ("Education",
               "Construction of buildings for creches and anganwadies",
               "3.15", "building", 1_150_000, (1, 3)),

    "4.001": ("Public health",
              "Construction of rooms and facilities for hospitals, FWC, PHC Centers and ANM centers",
              "4.1", "room", 980_000, (1, 4)),

    "10.001": ("Railways, roads, bridges and pathways",
               "Construction of roads, link roads, pathways or any other road with or without drainage system",
               "10.1", "km", 4_200_000, (0.4, 5.0)),

    "10.003": ("Railways, roads, bridges and pathways",
               "Construction of culverts and bridges",
               "10.3", "culvert", 1_800_000, (1, 4)),

    "13.001": ("Energy supply and distribution systems",
               "Street lights",
               "9.1", "pole", 28_000, (15, 180)),

    "12.001": ("Public recreational facilities, sports and parks",
               "Development of public parks",
               "12.1", "sqm", 2_600, (200, 2500)),

    "1.001": ("Public and Community Building",
              "Construction of community centers and community halls",
              "1.1", "sqm", 7_500, (80, 500)),
}

# ─────────────────────────────────────────────────────────────
# GEOGRAPHY
# ─────────────────────────────────────────────────────────────
# state -> district -> (constituency, [rural blocks], [urban wards])
GEOGRAPHY = {
    "Bihar": {
        "Nalanda":     ("NALANDA",     ["Rajgir", "Silao", "Islampur", "Hilsa", "Ekangarsarai"]),
        "Gaya":        ("GAYA",        ["Bodh Gaya", "Sherghati", "Tikari", "Belaganj", "Wazirganj"]),
        "Muzaffarpur": ("MUZAFFARPUR", ["Kanti", "Musahri", "Bochaha", "Sakra", "Kurhani"]),
        "Bhagalpur":   ("BHAGALPUR",   ["Naugachhia", "Kahalgaon", "Sultanganj", "Pirpainti"]),
    },
    "West Bengal": {
        "Nadia":       ("KRISHNANAGAR", ["Chakdaha", "Ranaghat", "Nabadwip", "Tehatta", "Kaliganj"]),
        "Hooghly":     ("HOOGHLY",      ["Arambagh", "Dhaniakhali", "Polba", "Goghat"]),
        "Murshidabad": ("BAHARAMPUR",   ["Beldanga", "Domkal", "Raninagar", "Hariharpara"]),
        "Howrah":      ("HOWRAH",       ["Bagnan", "Amta", "Uluberia", "Sankrail"]),
    },
    "Karnataka": {
        "Tumakuru":       ("TUMKUR",     ["Sira", "Madhugiri", "Kunigal", "Gubbi", "Pavagada"]),
        "Belagavi":       ("BELGAUM",    ["Bailhongal", "Saundatti", "Ramdurg", "Khanapur"]),
        "Mysuru":         ("MYSORE",     ["Nanjangud", "T Narasipura", "Hunsur", "Periyapatna"]),
        "Bengaluru Urban":("BANGALORE NORTH", ["Yelahanka", "Anekal", "Kengeri", "Jigani"]),
    },
    "Maharashtra": {
        "Pune":   ("PUNE",   ["Junnar", "Ambegaon", "Bhor", "Indapur", "Daund"]),
        "Nagpur": ("NAGPUR", ["Katol", "Ramtek", "Umred", "Kamptee"]),
        "Nashik": ("NASHIK", ["Sinnar", "Igatpuri", "Dindori", "Niphad"]),
        "Thane":  ("THANE",  ["Shahapur", "Murbad", "Bhiwandi", "Ambernath"]),
    },
}

# ─────────────────────────────────────────────────────────────
# IMPLEMENTING AGENCIES
# ─────────────────────────────────────────────────────────────
AGENCIES = [
    "PWD Division",
    "Zilla Parishad Works",
    "Rural Engineering Services",
    "Municipal Corporation Works",
    "Block Development Office",
    "Jal Nigam Unit",
    "Public Health Engineering Dept",
    "Panchayat Samiti Cell",
]

VENDORS = [
    "Shreeji Constructions", "Maa Durga Enterprises", "Balaji Infratech",
    "New Bharat Builders", "Sai Ram Constructions", "Ganesh Traders",
    "Vishwakarma Engineering", "Annapurna Infra", "Om Sai Contractors",
    "Jai Hind Construction Co", "Laxmi Narayan Works", "Sunrise Engineers",
]

# ─────────────────────────────────────────────────────────────
# DESCRIPTION TEMPLATES  (per PFMS code)
#
# Real descriptions name a place. That is exactly what makes two
# genuine works distinguishable from one work billed twice.
# ─────────────────────────────────────────────────────────────
DESCRIPTION_TEMPLATES = {
    "5.001":  ["Installation of borewell at {place}, {area}",
               "Drilling of tube-well near {place} in {area}",
               "Providing borewell facility at {place} {area}"],
    "5.002":  ["Installation of hand pumps at {place}, {area}",
               "Providing hand pumps near {place} in {area}"],
    "5.003":  ["Construction of overhead water tank at {place}, {area}",
               "Water storage tank near {place}, {area}"],
    "1.5.7":  ["Construction of pucca drain along {place} road, {area}",
               "Providing drainage line near {place} in {area}",
               "RCC drain construction at {place}, {area}"],
    "1.5.8":  ["Construction of public toilet block at {place}, {area}",
               "Public sanitation complex near {place}, {area}"],
    "3.001":  ["Construction of additional classrooms at Govt School, {place}, {area}",
               "School room construction at {place} in {area}"],
    "1.3.15": ["Construction of anganwadi building at {place}, {area}",
               "Creche building near {place}, {area}"],
    "4.001":  ["Construction of rooms at PHC {place}, {area}",
               "Additional ward rooms at health centre, {place}, {area}"],
    "10.001": ["Construction of link road from {place} to main road, {area}",
               "Bitumen road construction at {place} village, {area}",
               "Pathway construction near {place}, {area}"],
    "10.003": ["Construction of culvert on {place} nala, {area}",
               "Bridge construction near {place}, {area}"],
    "13.001": ["Installation of solar street lights at {place}, {area}",
               "Street lighting along {place} main road, {area}"],
    "12.001": ["Development of public park at {place}, {area}",
               "Park and green space near {place}, {area}"],
    "1.001":  ["Construction of community hall at {place}, {area}",
               "Community centre building near {place}, {area}"],
}

# Landmarks used to build place names inside descriptions
LANDMARKS = [
    "Primary School", "Panchayat Bhavan", "Bus Stand", "Main Chowk",
    "Health Centre", "Market Road", "Railway Crossing", "Temple Road",
    "Post Office", "Anganwadi Centre", "Police Chowki", "Water Tank",
    "High School", "Community Centre", "Ration Shop", "Bridge Point",
]


# ─────────────────────────────────────────────────────────────
# DELAY DRIVERS
#
# Duration in the first version of this generator was random(45, 400),
# independent of everything. A delay model trained on it scored AUC 0.503
# — a coin flip — because there was nothing to learn.
#
# These factors model delay mechanisms that are documented in Indian
# public works. They are assumptions about the world, not measurements,
# and any model trained on them MUST be re-validated on real data before
# it is trusted.
# ─────────────────────────────────────────────────────────────

# Typical build time by PFMS code, in days. A hand pump is not a bridge.
BASE_DURATION = {
    "5.001": 70,    "5.002": 45,    "5.003": 120,   "1.5.7": 150,
    "1.5.8": 130,   "3.001": 210,   "1.3.15": 240,  "4.001": 250,
    "10.001": 300,  "10.003": 280,  "13.001": 60,   "12.001": 170,
    "1.001": 230,
}

# Some implementing agencies are chronically slower than others —
# staffing, contractor pool, workload.
AGENCY_SPEED = {
    "PWD Division":                   0.92,
    "Zilla Parishad Works":           1.05,
    "Rural Engineering Services":     1.00,
    "Municipal Corporation Works":    0.88,
    "Block Development Office":       1.28,
    "Jal Nigam Unit":                 0.95,
    "Public Health Engineering Dept": 1.10,
    "Panchayat Samiti Cell":          1.35,
}

# Works sanctioned just before or during the monsoon start late and run
# long. June to September.
MONSOON_MONTHS = {6, 7, 8, 9}
MONSOON_FACTOR = 1.30

# Remote or hilly districts run slower.
DISTRICT_FACTOR = {
    "Gaya": 1.15, "Bhagalpur": 1.12, "Murshidabad": 1.10,
    "Belagavi": 1.08, "Nashik": 1.06, "Pavagada": 1.10,
}

# Bigger works take proportionally longer.
SIZE_FACTOR_PER_LOG = 0.18