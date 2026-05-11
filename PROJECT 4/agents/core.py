"""
agents/core.py  —  Jathakam AI Multi-Agent System
"""
from __future__ import annotations
import json, re, base64, textwrap
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

def _openai():
    import openai; return openai
def _pypdf():
    from pypdf import PdfReader; return PdfReader

class BaseAgent:
    NAME="BaseAgent"; EMOJI="🤖"; COLOR="#6c63ff"
    def __init__(self, api_key, model="gpt-4o-mini"):
        self._key=api_key; self.model=model; self.logs=[]
    def _chat(self, system, user, temperature=0.2, max_tokens=2000):
        client=_openai().OpenAI(api_key=self._key)
        r=client.chat.completions.create(model=self.model,temperature=temperature,max_tokens=max_tokens,
            messages=[{"role":"system","content":system},{"role":"user","content":user}])
        return r.choices[0].message.content
    def _vision(self, prompt, image_b64, media_type="image/jpeg"):
        client=_openai().OpenAI(api_key=self._key)
        r=client.chat.completions.create(model="gpt-4o",max_tokens=2000,
            messages=[{"role":"user","content":[{"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:{media_type};base64,{image_b64}","detail":"high"}}]}])
        return r.choices[0].message.content
    def _log(self,action,detail=""):
        e={"agent":self.NAME,"emoji":self.EMOJI,"action":action,"detail":detail,"ts":datetime.now().strftime("%H:%M:%S")}
        self.logs.append(e); return e

class OCRAgent(BaseAgent):
    NAME="OCRAgent"; EMOJI="👁️"; COLOR="#00c8ff"
    OCR_PROMPT="""You are an expert OCR system for Vedic astrology documents (jathakam/horoscope).
Extract ALL text from this image/document including:
- Malayalam script (ജാതകം, ലഗ്നം, നക്ഷത്രം, etc.)
- Sanskrit terms (Mesha, Surya, Chandra, Lagna, Nakshatra, etc.)
- Planet positions in houses (which planet is in which house/rasi)
- Personal details: name, date/time/place of birth
- Dasha periods, yogas, doshas mentioned
- Any tables or grid data showing planetary positions
Preserve structure. Output everything you can read."""

    def run(self, file_path):
        self._log("Starting OCR", Path(file_path).name)
        ext=Path(file_path).suffix.lower()
        if ext==".pdf":       return self._ocr_pdf(file_path)
        elif ext in(".jpg",".jpeg",".png",".webp",".bmp"):
                               return self._ocr_image(file_path,ext)
        elif ext in(".txt",".md"):
            t=open(file_path,encoding="utf-8",errors="replace").read()
            self._log("Text read",f"{len(t)} chars"); return t,"✅ Text file read"
        return "","❌ Unsupported format"

    def _ocr_pdf(self, path):
        try:
            PdfReader=_pypdf(); reader=PdfReader(path)
            txt="\n\n".join(p.extract_text() or "" for p in reader.pages).strip()
            if len(txt)>100:
                self._log("PDF text extracted",f"{len(txt)} chars")
                return txt,f"✅ PDF text extracted ({len(reader.pages)} pages)"
            return self._pdf_vision_ocr(path)
        except Exception as e: return "",f"❌ PDF error: {e}"

    def _pdf_vision_ocr(self, path):
        try:
            import fitz
            doc=fitz.open(path); pix=doc[0].get_pixmap(dpi=200)
            b64=base64.b64encode(pix.tobytes("jpeg")).decode()
            t=self._vision(self.OCR_PROMPT,b64,"image/jpeg")
            self._log("Vision OCR (PDF)",f"{len(t)} chars"); return t,"✅ Vision OCR on PDF"
        except ImportError:
            return "","⚠️ Install PyMuPDF for scanned PDFs: pip install pymupdf"
        except Exception as e: return "",f"❌ {e}"

    def _ocr_image(self, path, ext):
        self._log("GPT-4o vision OCR", Path(path).name)
        media_map={".jpg":"image/jpeg",".jpeg":"image/jpeg",".png":"image/png",".webp":"image/webp",".bmp":"image/bmp"}
        try:
            b64=base64.b64encode(open(path,"rb").read()).decode()
            t=self._vision(self.OCR_PROMPT,b64,media_map.get(ext,"image/jpeg"))
            self._log("Image OCR done",f"{len(t)} chars"); return t,"✅ Image OCR complete"
        except Exception as e: return "",f"❌ {e}"


class ExtractionAgent(BaseAgent):
    NAME="ExtractionAgent"; EMOJI="🔬"; COLOR="#00e5a0"
    SYSTEM="""You are a Vedic astrology data extractor. Extract structured JSON from jathakam text.
Handle Malayalam (ലഗ്നം=Lagna, രാശി=Rasi, നക്ഷത്രം=Nakshatra, സൂര്യൻ=Surya, ചന്ദ്രൻ=Chandra,
ചൊവ്വ=Kuja/Mars, ബുധൻ=Budha, വ്യാഴം=Guru, ശുക്രൻ=Shukra, ശനി=Shani, രാഹു=Rahu, കേതു=Ketu),
Sanskrit, and English terms. Rasi numbers: Mesha=1,Vrishabha=2,Mithuna=3,Kataka=4,Simha=5,Kanya=6,
Tula=7,Vrischika=8,Dhanus=9,Makara=10,Kumbha=11,Meena=12. Respond ONLY with valid JSON."""

    def run(self, raw_text):
        self._log("Extracting structure",f"{len(raw_text)} chars")
        schema={"personal":{"name":None,"date_of_birth":None,"time_of_birth":None,"place_of_birth":None,"gender":None},
                "lagna":{"rasi":None,"rasi_number":None,"degree":None},
                "rasi_moon_sign":{"rasi":None,"rasi_number":None},
                "nakshatra":{"name":None,"pada":None,"lord":None},
                "planets":{"Surya":{"house":None,"rasi":None,"retrograde":False},"Chandra":{"house":None,"rasi":None,"retrograde":False},
                           "Kuja":{"house":None,"rasi":None,"retrograde":False},"Budha":{"house":None,"rasi":None,"retrograde":False},
                           "Guru":{"house":None,"rasi":None,"retrograde":False},"Shukra":{"house":None,"rasi":None,"retrograde":False},
                           "Shani":{"house":None,"rasi":None,"retrograde":False},"Rahu":{"house":None,"rasi":None,"retrograde":False},
                           "Ketu":{"house":None,"rasi":None,"retrograde":False}},
                "dasha":{"current_mahadasha":None,"current_antardasha":None,"dasha_end":None},
                "special_notes":"","raw_language":""}
        prompt=f"Extract from this jathakam text into this JSON schema:\n{json.dumps(schema,indent=2)}\n\nText:\n{raw_text[:3000]}\n\nReturn ONLY valid JSON."
        raw=self._chat(self.SYSTEM,prompt,temperature=0.1,max_tokens=2000)
        try:
            data=json.loads(re.sub(r"```(?:json)?|```","",raw).strip())
            self._log("Extraction OK","✅"); return data,"✅ Structured extraction done"
        except Exception as e:
            self._log("Parse failed",str(e)[:60])
            return {"raw_extraction":raw,"parse_error":str(e)},f"⚠️ Partial: {e}"


class InterpretationAgent(BaseAgent):
    NAME="InterpretationAgent"; EMOJI="🔮"; COLOR="#c084fc"
    SYSTEM="""You are a master Vedic Jyotishi with decades of experience in Parashari astrology.
Write interpretations that are: warm, specific (name actual planets/houses), accessible to non-astrologers,
practically useful, and balanced (honest about challenges, positive about strengths).
Always note this is astrological tradition, not prediction."""

    def run(self, extracted, raw_text):
        self._log("Starting interpretation")
        lagna=extracted.get("lagna",{}); rasi=extracted.get("rasi_moon_sign",{})
        naksha=extracted.get("nakshatra",{}); planets=extracted.get("planets",{})
        dasha=extracted.get("dasha",{})
        ctx=f"""LAGNA: {lagna.get('rasi','?')} (House {lagna.get('rasi_number','?')})
MOON SIGN: {rasi.get('rasi','?')}  |  NAKSHATRA: {naksha.get('name','?')} Pada {naksha.get('pada','?')} (Lord: {naksha.get('lord','?')})
DASHA: {dasha.get('current_mahadasha','?')} / {dasha.get('current_antardasha','?')}
PLANETS:\n{self._fmt_planets(planets)}
NOTES: {extracted.get('special_notes','None')}"""
        results={}
        sections=[
            ("personality","150-200 word Personality Profile: core traits from Lagna, mental nature from Moon+Nakshatra, key strengths. Write in second person."),
            ("career","150-200 word Career Analysis: natural career fields from 10th house, skills from planetary positions, financial prospects, best professions."),
            ("yogas","Identify Special Yogas (Raja Yoga, Dhana Yoga, Gaja Kesari, Pancha Mahapurusha, etc.). For each: state if present, what it means simply, strength level."),
            ("dasha","Analyse Current Dasha Period: what the Mahadasha planet means, Antardasha themes, current life focus, favorable/challenging areas."),
            ("health","100-150 word Health Tendencies: body parts from Lagna, planetary health indicators, vitality level. Note: not medical advice."),
            ("relationships","100-150 word Relationships: Venus placement, 7th house condition, partner type indicated, marriage timing tendencies, family life."),
            ("doshas",None),
        ]
        for section,instr in sections:
            self._log(f"Generating {section}")
            if section=="doshas":
                results[section]=self._detect_doshas(ctx)
            else:
                prompt=f"Based on this jathakam:\n{ctx}\n\n{instr}"
                try: results[section]=self._chat(self.SYSTEM,prompt,temperature=0.4,max_tokens=500)
                except Exception as e: results[section]=f"_Error: {e}_"
        self._log("Interpretation done","✅"); return results

    def _fmt_planets(self, planets):
        lines=[]
        for p,info in planets.items():
            if isinstance(info,dict) and info.get("house"):
                r=f" (R)" if info.get("retrograde") else ""
                lines.append(f"  {p}: House {info['house']} {info.get('rasi','')}{r}")
        return "\n".join(lines) or "  (positions not fully extracted)"

    def _detect_doshas(self, ctx):
        prompt=f"""Based on this chart:\n{ctx}\n\nCheck for:
1. **Mangal Dosha**: Mars in houses 1,2,4,7,8,12 from Lagna/Moon/Venus?
2. **Kala Sarpa Dosha**: All 7 planets hemmed between Rahu-Ketu?
3. **Pitru Dosha**: Sun/9th house afflicted?
4. **Guru Chandala Yoga**: Jupiter conjunct Rahu?
For each: Present/Absent/Cannot determine. Brief explanation. 1-2 traditional remedies if present.
Be accurate. If data insufficient, say so."""
        try: return self._chat(self.SYSTEM,prompt,temperature=0.2,max_tokens=600)
        except Exception as e: return f"_Dosha analysis unavailable: {e}_"


class LanguageAgent(BaseAgent):
    NAME="LanguageAgent"; EMOJI="🌐"; COLOR="#ffb84d"
    SYSTEM="""You are an expert in Vedic astrology Sanskrit and Malayalam terminology.
Explain terms clearly, warmly, and accessibly for people with no astrology background.
Always give: original term, transliteration, English meaning, astrological significance."""

    def explain_term(self, term):
        try: return self._chat(self.SYSTEM,f"Explain this Vedic astrology term: '{term}'. Include meaning, significance, and practical relevance.",temperature=0.3,max_tokens=300)
        except Exception as e: return f"_Could not explain '{term}': {e}_"

    def run(self, raw_text):
        self._log("Language analysis")
        prompt=f"""Analyse this jathakam text. What languages/scripts are present?
List any Malayalam or Sanskrit technical terms found.
Text: {raw_text[:800]}
Respond in JSON: {{"languages":[],"terms_found":[{{"term":"","meaning":""}}],"summary":""}}"""
        try:
            raw=self._chat(self.SYSTEM,prompt,temperature=0.1,max_tokens=400)
            return json.loads(re.sub(r"```(?:json)?|```","",raw).strip())
        except Exception: return {"languages":["Unknown"],"terms_found":[],"summary":"Language detection failed"}


# Registry
AGENT_INFO=[
    {"name":"OCRAgent",            "emoji":"👁️","color":"#00c8ff","desc":"Extracts text from PDF/image using GPT-4o vision"},
    {"name":"ExtractionAgent",     "emoji":"🔬","color":"#00e5a0","desc":"Parses raw text → structured Lagna, planets, Nakshatra JSON"},
    {"name":"InterpretationAgent", "emoji":"🔮","color":"#c084fc","desc":"Generates personality, career, dosha, yoga analysis"},
    {"name":"LanguageAgent",       "emoji":"🌐","color":"#ffb84d","desc":"Malayalam/Sanskrit term translation & explanation"},
]