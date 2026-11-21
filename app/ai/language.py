"""
Language handling (Member 1) - R1 English / Hindi / Hinglish.

1. detect_language(): decides en / hi / hinglish with simple, explainable rules.
2. to_english_keywords(): turns Hindi / Hinglish shopping words into English
   search words ("sasta joota" -> "cheap shoes") so keyword search works.
3. phrase(): reply templates in all three languages (used in mock mode and
   for fixed messages such as refusals).
"""
import re
from typing import Dict, List

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_WORD = re.compile(r"[ऀ-ॿ]+|[a-zA-Z]+")

# Words that only appear in Romanised Hindi (Hinglish). Strong markers decide alone.
STRONG_HINGLISH = {
    "mujhe", "chahiye", "chahie", "sasta", "sasti", "saste", "dikhao", "dikha", "batao", "bata",
    "kitna", "kitne", "kitni", "wala", "wale", "wali", "kharida", "kharidna", "milega", "milegi",
    "kaunsa", "konsa", "kaun", "sabse", "accha", "acha", "achha", "chahta", "chahti", "karo",
    "hoga", "hogi", "kya", "nahi", "nahin", "joota", "joote", "jootey", "kapde", "liye",
}
WEAK_HINGLISH = {"hai", "hain", "ka", "ki", "ke", "mein", "aur", "ye", "yeh", "woh", "se", "ko",
                 "bhi", "ek", "koi", "kam", "zyada", "jyada", "papa", "bhai", "tak", "andar", "sirf",
                 "kuch", "karni", "karna", "lena", "leni", "dekhna", "hoon", "hu", "mai", "tha", "thi",
                 "chahte", "chahiye", "accha", "acha", "achha", "badiya", "wala", "wali"}

# Hindi / Hinglish -> English search words
TRANSLATIONS: Dict[str, str] = {
    # Hinglish
    "sasta": "cheap", "sasti": "cheap", "saste": "cheap", "joota": "shoes", "joote": "shoes",
    "jootey": "shoes", "jute": "shoes", "chappal": "sandals", "kapde": "clothes", "kameez": "shirt",
    "ghadi": "watch", "papa": "father", "pitaji": "father", "maa": "mother", "mummy": "mother",
    "bhai": "brother", "behen": "sister", "dost": "friend", "tohfa": "gift", "uphaar": "gift",
    "kala": "black", "kaala": "black", "neela": "blue", "laal": "red", "safed": "white",
    "hara": "green", "peela": "yellow", "garmi": "summer", "sardi": "winter", "khana": "cooking",
    "rasoi": "kitchen", "baal": "hair", "chehra": "face", "dhoop": "sunscreen",
    "chadar": "bedsheet", "chaddar": "bedsheet", "takiya": "pillow", "parda": "curtain", "parde": "curtains",
    "kalai": "wrist", "bartan": "utensils", "katori": "bowl", "thali": "plate", "gilas": "glass",
    "botal": "bottle", "kambal": "blanket", "tauliya": "towel", "kaleen": "carpet", "dari": "rug",
    "ghadiyan": "watch", "ghadiya": "watch", "mobile": "mobile", "kavar": "cover",
    # Devanagari
    "सस्ता": "cheap", "सस्ती": "cheap", "सस्ते": "cheap", "जूते": "shoes", "जूता": "shoes",
    "चप्पल": "sandals", "कपड़े": "clothes", "कमीज": "shirt", "कमीज़": "shirt", "घड़ी": "watch",
    "स्पीकर": "speaker", "ब्लूटूथ": "bluetooth", "फोन": "phone", "फ़ोन": "phone",
    "लैपटॉप": "laptop", "बैग": "bag", "कुकर": "cooker", "प्रेशर": "pressure", "टीशर्ट": "t-shirt",
    "टी-शर्ट": "t-shirt", "पिता": "father", "पापा": "father", "माँ": "mother", "उपहार": "gift",
    "तोहफा": "gift", "दौड़": "running", "दौड़ने": "running", "ट्रेकिंग": "trekking",
    "काला": "black", "नीला": "blue", "लाल": "red", "सफेद": "white", "हरा": "green",
    "फेस": "face", "वॉश": "wash", "शैम्पू": "shampoo", "क्रीम": "cream", "जूतों": "shoes",
    "स्पीकरों": "speaker", "हेडफोन": "headphones", "ईयरफोन": "earphones", "घड़ियाँ": "watch",
    "घड़ियां": "watch", "चादर": "bedsheet", "तकिया": "pillow", "पर्दा": "curtain", "पर्दे": "curtains",
    "कवर": "cover", "मोबाइल": "mobile", "चार्जर": "charger", "माउस": "mouse", "कीबोर्ड": "keyboard",
    "मग": "mug", "बोतल": "bottle", "गिलास": "glass", "कप": "cup", "कटोरी": "bowl", "थाली": "plate",
    "बर्तन": "utensils", "तौलिया": "towel", "कालीन": "carpet", "कंबल": "blanket", "कुशन": "cushion",
    "पाउच": "pouch", "राउटर": "router", "पेन": "pen", "ड्राइव": "drive", "लिपस्टिक": "lipstick",
    "परफ्यूम": "perfume", "डियोडरेंट": "deodorant", "तेल": "oil", "बाल": "hair", "रंग": "color",
}

# Fixed reply templates. {placeholders} are filled by code with catalogue facts.
PHRASES: Dict[str, Dict[str, str]] = {
    "found": {
        "en": "Here {verb} {n} option{s} from our catalogue{extra}:",
        "hinglish": "Ye rahe {n} option{s} hamare catalogue se{extra}:",
        "hi": "हमारे कैटलॉग से {n} विकल्प{extra}:",
    },
    "not_found": {
        "en": "I couldn't find that in our catalogue{why}. Would you like me to widen the search?",
        "hinglish": "Ye hamare catalogue mein nahi mila{why}. Kya main search thoda badha doon?",
        "hi": "यह हमारे कैटलॉग में नहीं मिला{why}। क्या मैं खोज थोड़ी बढ़ा दूँ?",
    },
    "out_of_scope": {
        "en": "I couldn't find that in our catalogue - we sell {categories}.",
        "hinglish": "Ye hamare catalogue mein nahi hai - hum ye categories bechte hain: {categories}.",
        "hi": "यह हमारे कैटलॉग में नहीं है - हमारी श्रेणियाँ: {categories}।",
    },
    "privacy": {
        "en": "Sorry, I can't share another shopper's information. I can only help with your own orders and our catalogue.",
        "hinglish": "Sorry, main kisi doosre shopper ki jaankari share nahi kar sakta. Main sirf aapke apne orders aur catalogue mein madad kar sakta hoon.",
        "hi": "क्षमा करें, मैं किसी दूसरे ग्राहक की जानकारी साझा नहीं कर सकता। मैं केवल आपके अपने ऑर्डर और कैटलॉग में मदद कर सकता हूँ।",
    },
    "injection": {
        "en": "I can't change how I work based on instructions in a message. I'm happy to help you find products from our catalogue.",
        "hinglish": "Main message ke andar diye instructions follow nahi kar sakta. Catalogue se product dhoondhne mein zaroor madad karunga.",
        "hi": "मैं संदेश में दिए गए निर्देशों के आधार पर अपना काम नहीं बदल सकता। कैटलॉग से उत्पाद खोजने में ज़रूर मदद करूँगा।",
    },
    "session_owner": {
        "en": "This chat session belongs to a different shopper. Please start a new chat.",
        "hinglish": "Ye chat session kisi aur shopper ka hai. Kripya nayi chat shuru karein.",
        "hi": "यह चैट सत्र किसी दूसरे ग्राहक का है। कृपया नई चैट शुरू करें।",
    },
    "greeting": {
        "en": "Hi! What are you shopping for today?",
        "hinglish": "Namaste! Aaj aap kya dhoondh rahe hain?",
        "hi": "नमस्ते! आज आप क्या खरीदना चाहते हैं?",
    },
    "error": {
        "en": "Sorry, something went wrong on my side. Please try again.",
        "hinglish": "Sorry, kuch gadbad ho gayi. Kripya dobara try karein.",
        "hi": "क्षमा करें, कुछ गड़बड़ हो गई। कृपया दोबारा कोशिश करें।",
    },
    "compare_need_two": {
        "en": "Which products should I compare? Tell me two product IDs, or search first and say 'compare the first two'.",
        "hinglish": "Kaun se products compare karoon? Do product ID bataiye, ya pehle search karke 'pehle do compare karo' boliye.",
        "hi": "किन उत्पादों की तुलना करूँ? दो प्रोडक्ट ID बताइए, या पहले खोजकर 'पहले दो की तुलना करो' कहिए।",
    },
    "which_product": {
        "en": "Which product do you mean? Please share the product ID or search for it first.",
        "hinglish": "Aap kis product ki baat kar rahe hain? Product ID bataiye ya pehle search kijiye.",
        "hi": "आप किस उत्पाद की बात कर रहे हैं? प्रोडक्ट ID बताइए या पहले खोजिए।",
    },
    "compare_intro": {
        "en": "Here is a side-by-side comparison from our catalogue.",
        "hinglish": "Ye raha catalogue se side-by-side comparison.",
        "hi": "कैटलॉग से तुलना यह रही।",
    },
    "compare_missing": {
        "en": "I can only compare products we actually list. Not in our catalogue: {missing}. I won't guess details for products we don't sell.",
        "hinglish": "Main sirf hamare catalogue ke products compare kar sakta hoon. Catalogue mein nahi hai: {missing}. Jo hum nahi bechte, uski details main guess nahi karunga.",
        "hi": "मैं केवल हमारे कैटलॉग के उत्पादों की तुलना कर सकता हूँ। कैटलॉग में नहीं है: {missing}। जो हम नहीं बेचते, उसकी जानकारी का अनुमान नहीं लगाऊँगा।",
    },
    "synthetic_orders": {
        "en": "(Order history is synthetic demo data.)",
        "hinglish": "(Order history synthetic demo data hai.)",
        "hi": "(ऑर्डर इतिहास सिंथेटिक डेमो डेटा है।)",
    },
    "no_orders": {
        "en": "You don't have any past orders with us yet.",
        "hinglish": "Aapka abhi tak koi purana order nahi hai.",
        "hi": "आपका अभी तक कोई पिछला ऑर्डर नहीं है।",
    },
    "your_orders": {
        "en": "Your recent orders:",
        "hinglish": "Aapke recent orders:",
        "hi": "आपके हाल के ऑर्डर:",
    },
}

CLARIFY_QUESTIONS: Dict[str, Dict[str, str]] = {
    "product_type": {
        "en": "What kind of product are you looking for - for example a phone cover, a wrist watch, a bedsheet or kitchenware?",
        "hinglish": "Aap kis type ka product dhoondh rahe hain - jaise phone cover, ghadi, bedsheet ya kitchen ka saamaan?",
        "hi": "आप किस तरह का उत्पाद ढूँढ रहे हैं - जैसे मोबाइल कवर, घड़ी, चादर या रसोई का सामान?",
    },
    "budget": {
        "en": "What's your budget?",
        "hinglish": "Aapka budget kitna hai?",
        "hi": "आपका बजट कितना है?",
    },
    "budget_use": {
        "en": "What's your budget, and what will you mostly use it for{examples}?",
        "hinglish": "Aapka budget kitna hai, aur zyada tar kis kaam ke liye chahiye{examples}?",
        "hi": "आपका बजट कितना है, और इसका ज़्यादातर उपयोग किस काम के लिए होगा{examples}?",
    },
    "recipient": {
        "en": "Who is the gift for, and what do they enjoy?",
        "hinglish": "Gift kiske liye hai, aur unhe kya pasand hai?",
        "hi": "उपहार किसके लिए है, और उन्हें क्या पसंद है?",
    },
    "use": {
        "en": "What will you mostly use it for{examples}?",
        "hinglish": "Zyada tar kis kaam ke liye chahiye{examples}?",
        "hi": "इसका ज़्यादातर उपयोग किस काम के लिए होगा{examples}?",
    },
}

SUGGESTIONS: Dict[str, Dict[str, str]] = {
    "only_brand": {"en": "Only {brand}", "hinglish": "Sirf {brand} dikhao", "hi": "सिर्फ {brand} दिखाओ"},
    "cheaper": {"en": "Show cheaper ones", "hinglish": "Aur saste dikhao", "hi": "और सस्ते दिखाओ"},
    "compare": {"en": "Compare the first two", "hinglish": "Pehle do compare karo", "hi": "पहले दो की तुलना करो"},
    "under": {"en": "Under ₹{amount}", "hinglish": "₹{amount} se kam", "hi": "₹{amount} से कम"},
    "widen_budget": {"en": "Under ₹{amount}", "hinglish": "₹{amount} se kam", "hi": "₹{amount} से कम"},
    "any_brand": {"en": "Any brand", "hinglish": "Koi bhi brand", "hi": "कोई भी ब्रांड"},
}


def detect_language(text: str) -> str:
    """'hi' if mostly Devanagari, 'hinglish' if Romanised Hindi words appear, else 'en'."""
    if not text:
        return "en"
    letters = [c for c in text if c.isalpha()]
    devanagari = len(_DEVANAGARI.findall(text))
    if letters and devanagari / len(letters) > 0.3:
        return "hi"
    words = {w.lower() for w in re.findall(r"[a-zA-Z]+", text)}
    strong = len(words & STRONG_HINGLISH)
    weak = len(words & WEAK_HINGLISH)
    if strong >= 1 or weak >= 3:
        return "hinglish"
    return "en"


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def to_english_keywords(text: str) -> List[str]:
    """Translate known Hindi/Hinglish shopping words; keep English words as they are."""
    out: List[str] = []
    for word in tokenize(text):
        if word in TRANSLATIONS:
            out.append(TRANSLATIONS[word])
        elif _DEVANAGARI.search(word):
            continue                      # unknown Hindi word: skip for keyword search
        elif word in STRONG_HINGLISH or word in WEAK_HINGLISH:
            continue                      # grammar words carry no product meaning
        else:
            out.append(word)
    return out


def phrase(key: str, language: str, **values) -> str:
    table = PHRASES[key]
    template = table.get(language) or table["en"]
    return template.format(**values)


def clarify_question(slot: str, language: str, **values) -> str:
    table = CLARIFY_QUESTIONS[slot]
    template = table.get(language) or table["en"]
    return template.format(**{"examples": "", **values})


def suggestion(key: str, language: str, **values) -> str:
    table = SUGGESTIONS[key]
    template = table.get(language) or table["en"]
    return template.format(**values)
