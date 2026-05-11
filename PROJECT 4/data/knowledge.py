"""
data/knowledge.py
═══════════════════════════════════════════════════════
Embedded Vedic Astrology Knowledge Base
Used by RAG engine for grounded, accurate interpretations
═══════════════════════════════════════════════════════
"""

RASHIS = {
    "Mesha":    {"english": "Aries",       "number": 1,  "element": "Fire",  "lord": "Mars",    "quality": "Cardinal", "symbol": "♈"},
    "Vrishabha":{"english": "Taurus",      "number": 2,  "element": "Earth", "lord": "Venus",   "quality": "Fixed",    "symbol": "♉"},
    "Mithuna":  {"english": "Gemini",      "number": 3,  "element": "Air",   "lord": "Mercury", "quality": "Mutable",  "symbol": "♊"},
    "Kataka":   {"english": "Cancer",      "number": 4,  "element": "Water", "lord": "Moon",    "quality": "Cardinal", "symbol": "♋"},
    "Simha":    {"english": "Leo",         "number": 5,  "element": "Fire",  "lord": "Sun",     "quality": "Fixed",    "symbol": "♌"},
    "Kanya":    {"english": "Virgo",       "number": 6,  "element": "Earth", "lord": "Mercury", "quality": "Mutable",  "symbol": "♍"},
    "Tula":     {"english": "Libra",       "number": 7,  "element": "Air",   "lord": "Venus",   "quality": "Cardinal", "symbol": "♎"},
    "Vrischika":{"english": "Scorpio",     "number": 8,  "element": "Water", "lord": "Mars",    "quality": "Fixed",    "symbol": "♏"},
    "Dhanus":   {"english": "Sagittarius", "number": 9,  "element": "Fire",  "lord": "Jupiter", "quality": "Mutable",  "symbol": "♐"},
    "Makara":   {"english": "Capricorn",   "number": 10, "element": "Earth", "lord": "Saturn",  "quality": "Cardinal", "symbol": "♑"},
    "Kumbha":   {"english": "Aquarius",    "number": 11, "element": "Air",   "lord": "Saturn",  "quality": "Fixed",    "symbol": "♒"},
    "Meena":    {"english": "Pisces",      "number": 12, "element": "Water", "lord": "Jupiter", "quality": "Mutable",  "symbol": "♓"},
}

GRAHAS = {
    "Surya":    {"english": "Sun",     "symbol": "☉", "nature": "Malefic",  "color": "#FFB800", "signifies": "Soul, father, authority, government, health, ego"},
    "Chandra":  {"english": "Moon",    "symbol": "☽", "nature": "Benefic",  "color": "#C8E6FF", "signifies": "Mind, mother, emotions, intuition, public, travel"},
    "Kuja":     {"english": "Mars",    "symbol": "♂", "nature": "Malefic",  "color": "#FF4444", "signifies": "Energy, siblings, property, courage, accidents, surgery"},
    "Budha":    {"english": "Mercury", "symbol": "☿", "nature": "Neutral",  "color": "#00E5A0", "signifies": "Intelligence, communication, business, education, skin"},
    "Guru":     {"english": "Jupiter", "symbol": "♃", "nature": "Benefic",  "color": "#FFD700", "signifies": "Wisdom, wealth, children, religion, law, expansion"},
    "Shukra":   {"english": "Venus",   "symbol": "♀", "nature": "Benefic",  "color": "#FF80AB", "signifies": "Love, beauty, marriage, arts, luxury, vehicles"},
    "Shani":    {"english": "Saturn",  "symbol": "♄", "nature": "Malefic",  "color": "#8888FF", "signifies": "Discipline, karma, longevity, delays, servants, oil"},
    "Rahu":     {"english": "Rahu",    "symbol": "☊", "nature": "Malefic",  "color": "#AA88FF", "signifies": "Illusion, foreign, technology, unconventional, obsession"},
    "Ketu":     {"english": "Ketu",    "symbol": "☋", "nature": "Malefic",  "color": "#FF8844", "signifies": "Spirituality, past karma, moksha, isolation, research"},
}

NAKSHATRAS = [
    {"name": "Ashwini",     "lord": "Ketu",    "rasi": "Mesha",     "pada": [1,2,3,4], "quality": "Ashwa (Horse)", "deity": "Ashwini Kumaras"},
    {"name": "Bharani",     "lord": "Shukra",  "rasi": "Mesha",     "pada": [1,2,3,4], "quality": "Yoni (Womb)",   "deity": "Yama"},
    {"name": "Krittika",    "lord": "Surya",   "rasi": "Mesha/Vrishabha","pada":[1,2,3,4],"quality":"Razor",        "deity": "Agni"},
    {"name": "Rohini",      "lord": "Chandra", "rasi": "Vrishabha", "pada": [1,2,3,4], "quality": "Cart",          "deity": "Brahma"},
    {"name": "Mrigashira",  "lord": "Kuja",    "rasi": "Vrishabha/Mithuna","pada":[1,2,3,4],"quality":"Deer Head", "deity": "Soma"},
    {"name": "Ardra",       "lord": "Rahu",    "rasi": "Mithuna",   "pada": [1,2,3,4], "quality": "Teardrop",      "deity": "Rudra"},
    {"name": "Punarvasu",   "lord": "Guru",    "rasi": "Mithuna/Kataka","pada":[1,2,3,4],"quality":"Quiver",       "deity": "Aditi"},
    {"name": "Pushya",      "lord": "Shani",   "rasi": "Kataka",    "pada": [1,2,3,4], "quality": "Flower",        "deity": "Brihaspati"},
    {"name": "Ashlesha",    "lord": "Budha",   "rasi": "Kataka",    "pada": [1,2,3,4], "quality": "Serpent",       "deity": "Nagas"},
    {"name": "Magha",       "lord": "Ketu",    "rasi": "Simha",     "pada": [1,2,3,4], "quality": "Throne",        "deity": "Pitrs"},
    {"name": "Purva Phalguni","lord":"Shukra", "rasi": "Simha",     "pada": [1,2,3,4], "quality": "Hammock",       "deity": "Bhaga"},
    {"name": "Uttara Phalguni","lord":"Surya", "rasi": "Simha/Kanya","pada":[1,2,3,4], "quality": "Bed",           "deity": "Aryaman"},
    {"name": "Hasta",       "lord": "Chandra", "rasi": "Kanya",     "pada": [1,2,3,4], "quality": "Hand",          "deity": "Savitar"},
    {"name": "Chitra",      "lord": "Kuja",    "rasi": "Kanya/Tula","pada": [1,2,3,4], "quality": "Pearl",         "deity": "Tvashtr"},
    {"name": "Swati",       "lord": "Rahu",    "rasi": "Tula",      "pada": [1,2,3,4], "quality": "Coral",         "deity": "Vayu"},
    {"name": "Vishakha",    "lord": "Guru",    "rasi": "Tula/Vrischika","pada":[1,2,3,4],"quality":"Triumphal Arch","deity":"Indra-Agni"},
    {"name": "Anuradha",    "lord": "Shani",   "rasi": "Vrischika", "pada": [1,2,3,4], "quality": "Lotus",         "deity": "Mitra"},
    {"name": "Jyeshtha",    "lord": "Budha",   "rasi": "Vrischika", "pada": [1,2,3,4], "quality": "Earring",       "deity": "Indra"},
    {"name": "Mula",        "lord": "Ketu",    "rasi": "Dhanus",    "pada": [1,2,3,4], "quality": "Tied Bunch",    "deity": "Nirrti"},
    {"name": "Purva Ashadha","lord":"Shukra",  "rasi": "Dhanus",    "pada": [1,2,3,4], "quality": "Elephant Tusk", "deity": "Apas"},
    {"name": "Uttara Ashadha","lord":"Surya",  "rasi": "Dhanus/Makara","pada":[1,2,3,4],"quality":"Elephant Tusk", "deity": "Vishvadevas"},
    {"name": "Shravana",    "lord": "Chandra", "rasi": "Makara",    "pada": [1,2,3,4], "quality": "Ear",           "deity": "Vishnu"},
    {"name": "Dhanishtha",  "lord": "Kuja",    "rasi": "Makara/Kumbha","pada":[1,2,3,4],"quality":"Drum",         "deity": "Ashta Vasus"},
    {"name": "Shatabhisha", "lord": "Rahu",    "rasi": "Kumbha",    "pada": [1,2,3,4], "quality": "Empty Circle",  "deity": "Varuna"},
    {"name": "Purva Bhadra","lord": "Guru",    "rasi": "Kumbha/Meena","pada":[1,2,3,4],"quality":"Sword",         "deity": "Aja Ekapada"},
    {"name": "Uttara Bhadra","lord":"Shani",   "rasi": "Meena",     "pada": [1,2,3,4], "quality": "Twins",         "deity": "Ahir Budhnya"},
    {"name": "Revati",      "lord": "Budha",   "rasi": "Meena",     "pada": [1,2,3,4], "quality": "Fish",          "deity": "Pushan"},
]

HOUSES = {
    1:  {"name": "Lagna / Tanu",      "governs": "Self, appearance, personality, health, general life direction"},
    2:  {"name": "Dhana / Kutumba",   "governs": "Wealth, family, speech, food, face, values, savings"},
    3:  {"name": "Sahaja / Parakrama","governs": "Siblings, courage, communication, short travel, skills, efforts"},
    4:  {"name": "Sukha / Matru",     "governs": "Mother, home, happiness, vehicles, real estate, education, heart"},
    5:  {"name": "Putra / Buddhi",    "governs": "Children, intelligence, romance, speculation, past life merit, creativity"},
    6:  {"name": "Shatru / Roga",     "governs": "Enemies, diseases, debts, servants, legal disputes, daily work, obstacles"},
    7:  {"name": "Kalatra / Vivaha",  "governs": "Marriage, partnerships, business partners, public relations, spouse"},
    8:  {"name": "Ayu / Mrityu",      "governs": "Longevity, transformation, hidden matters, inheritance, occult, research"},
    9:  {"name": "Dharma / Bhagya",   "governs": "Luck, religion, father, higher education, long journeys, philosophy, guru"},
    10: {"name": "Karma / Rajya",     "governs": "Career, profession, reputation, government, authority, social status"},
    11: {"name": "Labha / Ayaya",     "governs": "Gains, income, elder siblings, friends, networks, wishes fulfilled"},
    12: {"name": "Vyaya / Moksha",    "governs": "Losses, expenditure, foreign lands, spirituality, sleep, moksha, hospitals"},
}

DOSHAS = {
    "Mangal Dosha": {
        "description": "Mars (Kuja) placed in houses 1, 2, 4, 7, 8, or 12 from Lagna, Moon, or Venus.",
        "effects": "May cause difficulties in marriage, spouse health issues, delays in marriage.",
        "remedies": ["Marrying someone with same dosha", "Kuja Dosha Nivaran Puja", "Wearing red coral after consultation"],
        "cancellations": ["If Mars is in own sign (Aries/Scorpio)", "If Mars is exalted (Capricorn)", "Jupiter aspects Mars", "Mars in 2nd house for Gemini/Virgo lagna"]
    },
    "Kala Sarpa Dosha": {
        "description": "All 7 planets (Sun to Saturn) hemmed between Rahu and Ketu.",
        "effects": "Obstacles in life, delays in achievements, struggles, but eventual success possible.",
        "remedies": ["Kala Sarpa Dosha Nivaran Puja", "Naga Pratishtha", "Visiting Trimbakeshwar/Ujjain"],
        "cancellations": ["Any planet outside Rahu-Ketu axis breaks the dosha", "Strong 5th/9th lords"]
    },
    "Pitru Dosha": {
        "description": "Sun or Moon afflicted by Rahu/Ketu/Saturn, or malefics in 9th house.",
        "effects": "Ancestral karma effects, obstacles from unexpected sources.",
        "remedies": ["Pitru Tarpanam", "Shraddha rituals on Amavasya", "Donating food to Brahmins"],
        "cancellations": []
    },
    "Shani Dosha (Sade Sati)": {
        "description": "Saturn transiting through 12th, 1st, and 2nd house from natal Moon.",
        "effects": "7.5 year period of challenges, introspection, karma clearing.",
        "remedies": ["Shani Puja every Saturday", "Hanuman Chalisa recitation", "Donating black sesame/mustard oil"],
        "cancellations": []
    }
}

PLANET_IN_SIGN_EFFECTS = {
    "Surya": {
        "Mesha": "Sun exalted — powerful, confident, leadership qualities, successful in career",
        "Simha": "Sun in own sign — authoritative, creative, generous, natural leader",
        "Tula": "Sun debilitated — may face authority issues, career struggles, needs humility",
        "default": "Strong sense of self, vitality depends on sign strength"
    },
    "Chandra": {
        "Vrishabha": "Moon exalted — very emotional intelligence, nurturing, popular, artistic",
        "Kataka": "Moon in own sign — intuitive, caring, good memory, home-loving",
        "Vrischika": "Moon debilitated — emotional turbulence, intense feelings, secretive",
        "default": "Emotional nature shaped by sign element and lord"
    },
    "Kuja": {
        "Makara": "Mars exalted — disciplined energy, excellent executive ability, achiever",
        "Mesha": "Mars in own sign — courageous, impulsive, athletic, independent",
        "Vrischika": "Mars in own sign — intense, investigative, transformative, determined",
        "Kataka": "Mars debilitated — scattered energy, emotional conflicts, needs grounding",
        "default": "Energy and ambition shaped by sign placement"
    },
    "Guru": {
        "Kataka": "Jupiter exalted — extremely wise, spiritual, fortunate, protector",
        "Dhanus": "Jupiter in own sign — philosophical, optimistic, excellent teacher",
        "Meena": "Jupiter in own sign — compassionate, spiritual, intuitive, healing",
        "Makara": "Jupiter debilitated — wisdom tested, material vs spiritual conflict",
        "default": "Wisdom and blessings expressed through sign quality"
    },
    "Shukra": {
        "Meena": "Venus exalted — artistic genius, deep love, spiritual romance, beauty",
        "Vrishabha": "Venus in own sign — sensual, artistic, love of luxury, charming",
        "Tula": "Venus in own sign — balanced in relationships, diplomatic, social",
        "Kanya": "Venus debilitated — critical in love, perfectionist in relationships",
        "default": "Love and aesthetics colored by sign"
    },
    "Shani": {
        "Tula": "Saturn exalted — exceptional discipline, justice-oriented, slow but sure success",
        "Makara": "Saturn in own sign — ambitious, structured, excellent administrator",
        "Kumbha": "Saturn in own sign — humanitarian, scientific, community-focused",
        "Mesha": "Saturn debilitated — impatience with discipline, karma around aggression",
        "default": "Karmic lessons and discipline through sign"
    },
}

CAREER_INDICATORS = """
Career Analysis in Vedic Astrology:

10th House Lord's Position:
- 10th lord in 1st: Self-employment, fame, personal brand
- 10th lord in 2nd: Finance, banking, family business, speaking
- 10th lord in 3rd: Media, communication, writing, arts, siblings' help
- 10th lord in 4th: Real estate, agriculture, vehicles, working from home
- 10th lord in 5th: Education, entertainment, creativity, children-related
- 10th lord in 6th: Medicine, legal, military, service industry
- 10th lord in 7th: Partnership business, diplomacy, public relations
- 10th lord in 8th: Research, occult, insurance, hidden professions
- 10th lord in 9th: Law, religion, higher education, international work
- 10th lord in 10th: Excellent career, leadership, government positions
- 10th lord in 11th: Multiple income sources, networking, gains from career
- 10th lord in 12th: Foreign work, hospitals, research, behind-the-scenes

Planets in 10th House:
- Sun: Government, authority, politics, administration
- Moon: Hospitality, public, nursing, import-export
- Mars: Military, engineering, surgery, real estate, sports
- Mercury: Business, writing, IT, communication, accounting
- Jupiter: Teaching, law, religion, finance, advisory
- Venus: Arts, entertainment, beauty, luxury goods, design
- Saturn: Industry, labor management, agriculture, mining
- Rahu: Foreign companies, technology, unconventional careers
- Ketu: Research, spirituality, behind-scenes technical roles
"""

PERSONALITY_BY_LAGNA = {
    "Mesha":     "Bold, pioneering, impulsive, competitive, natural leader, quick-tempered but forgiving. Athletic build, reddish complexion, leadership aura.",
    "Vrishabha": "Patient, sensual, stubborn, reliable, artistic, love of comfort and beauty. Well-built, pleasant voice, love of food and nature.",
    "Mithuna":   "Versatile, communicative, curious, witty, dual nature, adaptable. Tall, youthful appearance, expressive face, quick hands.",
    "Kataka":    "Nurturing, emotional, intuitive, protective, home-loving, mood swings. Round face, caring eyes, motherly demeanor, retentive memory.",
    "Simha":     "Proud, generous, creative, dramatic, leader, spotlight-seeker. Strong physique, commanding presence, warm heart, royal bearing.",
    "Kanya":     "Analytical, practical, service-oriented, perfectionist, health-conscious. Medium height, precise speech, discriminating mind, skillful hands.",
    "Tula":      "Diplomatic, charming, balanced, justice-seeking, indecisive. Attractive appearance, symmetrical features, artistic taste, social grace.",
    "Vrischika": "Intense, secretive, transformative, investigative, magnetic. Penetrating eyes, powerful presence, excellent detective skills, regenerative.",
    "Dhanus":    "Philosophical, optimistic, adventurous, blunt, freedom-loving. Athletic, tall, cheerful expression, love of travel and religion.",
    "Makara":    "Ambitious, disciplined, practical, responsible, slow but steady. Conservative appearance, prominent nose, serious demeanor, good bones.",
    "Kumbha":    "Humanitarian, innovative, independent, eccentric, futuristic. Unique appearance, friendly yet detached, scientific mind, social reformer.",
    "Meena":     "Compassionate, spiritual, intuitive, dreamy, empathetic. Soft features, large eyes, gentle demeanor, psychic tendencies, artistic.",
}

ASTROLOGY_KNOWLEDGE_CORPUS = f"""
# VEDIC ASTROLOGY COMPLETE KNOWLEDGE BASE

## THE 12 RASHIS (ZODIAC SIGNS)
{chr(10).join(f"**{r}** ({d['english']}): Element={d['element']}, Lord={d['lord']}, Quality={d['quality']}" for r, d in RASHIS.items())}

## THE 9 GRAHAS (PLANETS)
{chr(10).join(f"**{g}** ({d['english']}): Nature={d['nature']}, Signifies: {d['signifies']}" for g, d in GRAHAS.items())}

## THE 12 BHAVAS (HOUSES)
{chr(10).join(f"**House {h}** - {d['name']}: Governs {d['governs']}" for h, d in HOUSES.items())}

## THE 27 NAKSHATRAS
{chr(10).join(f"**{n['name']}**: Lord={n['lord']}, Rasi={n['rasi']}, Deity={n['deity']}" for n in NAKSHATRAS)}

## PERSONALITY BY LAGNA
{chr(10).join(f"**{l} Lagna**: {desc}" for l, desc in PERSONALITY_BY_LAGNA.items())}

## DOSHAS
{chr(10).join(f"**{d}**: {info['description']} Effects: {info['effects']}" for d, info in DOSHAS.items())}

## CAREER INDICATORS
{CAREER_INDICATORS}

## PLANET STRENGTH IN SIGNS
Exaltations: Sun in Aries, Moon in Taurus, Mars in Capricorn, Mercury in Virgo, Jupiter in Cancer, Venus in Pisces, Saturn in Libra.
Debilitations: Sun in Libra, Moon in Scorpio, Mars in Cancer, Mercury in Pisces, Jupiter in Capricorn, Venus in Virgo, Saturn in Aries.
Rahu is strong in Gemini/Taurus/Virgo. Ketu is strong in Sagittarius/Scorpio/Pisces.

## YOGA (SPECIAL COMBINATIONS)
- **Raja Yoga**: Lords of 1st and 5th/9th conjunct or in mutual aspect — authority, success, leadership.
- **Dhana Yoga**: Lords of 2nd and 11th together — wealth accumulation.
- **Gaja Kesari Yoga**: Jupiter in kendra from Moon — wisdom, fame, high status.
- **Pancha Mahapurusha Yoga**: A planet (except Sun/Moon) in own or exalted sign in kendra — exceptional person.
- **Budha Aditya Yoga**: Sun and Mercury together — intelligence, good communication, respected.
- **Chandra Mangal Yoga**: Moon and Mars together — wealth through mother or property.
- **Hamsa Yoga**: Jupiter in own/exalted sign in kendra — wisdom, spirituality, teaching.
- **Malavya Yoga**: Venus in own/exalted sign in kendra — beauty, artistic success, luxury.
- **Ruchaka Yoga**: Mars in own/exalted sign in kendra — military success, property, courage.
- **Shasha Yoga**: Saturn in own/exalted sign in kendra — discipline, service, long life.

## MALAYALAM ASTROLOGICAL TERMS
- ജാതകം (Jathakam) = Horoscope/Birth chart
- ലഗ്നം (Lagnam) = Ascendant/Rising sign
- രാശി (Rashi) = Zodiac sign/Moon sign
- നക്ഷത്രം (Nakshathram) = Birth star
- ഭാവം (Bhavam) = House
- ഗ്രഹം (Graham) = Planet
- ദോഷം (Dosham) = Malefic condition
- ദശ (Dasha) = Planetary period
- ഭഗ്യം (Bhagyam) = Fortune/Luck
- കർമ്മം (Karmam) = Work/Karma
"""