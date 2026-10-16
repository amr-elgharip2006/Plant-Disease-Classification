// General, non-prescriptive care tips per disease type. Keyed by PlantVillage
// folder name via CLASS_ADVICE. Always paired with the "consult an agronomist"
// disclaimer in the UI.

export const ADVICE = {
  healthy: {
    ar: [
      "النبات يبدو سليمًا. استمر على الري المنتظم والتسميد المتوازن.",
      "افحص الأوراق مرة كل أسبوع، خصوصًا الوجه السفلي.",
      "حافظ على مسافات كافية بين النباتات لتحسين التهوية.",
    ],
    en: [
      "The plant looks healthy. Keep regular watering and balanced fertilization.",
      "Check the leaves weekly, especially the undersides.",
      "Keep enough spacing between plants for good airflow.",
    ],
  },
  fungal: {
    ar: [
      "أزل الأوراق المصابة وتخلّص منها بعيدًا عن الحقل، ولا تضعها في السماد.",
      "اسقِ عند قاعدة النبات في الصباح وتجنّب تبليل الأوراق.",
      "حسّن التهوية بالتقليم وزيادة المسافات بين النباتات.",
      "طبّق الدورة الزراعية، واستشر مهندسًا زراعيًا لاختيار مبيد فطري مناسب إذا انتشرت الإصابة.",
    ],
    en: [
      "Remove infected leaves and dispose of them away from the field (not in compost).",
      "Water at the base in the morning and avoid wetting the leaves.",
      "Improve airflow by pruning and spacing plants further apart.",
      "Rotate crops, and ask an agronomist about a suitable fungicide if it spreads.",
    ],
  },
  lateBlight: {
    ar: [
      "مرض سريع الانتشار، خصوصًا في الجو الرطب والبارد. تصرّف فورًا.",
      "اقلع النباتات شديدة الإصابة وأتلفها، ولا تتركها في الحقل.",
      "أوقف الري بالرش وحافظ على جفاف الأوراق.",
      "استشر مهندسًا زراعيًا سريعًا بخصوص مبيد وقائي للنباتات السليمة المجاورة.",
    ],
    en: [
      "Spreads fast, especially in cool, humid weather. Act right away.",
      "Pull out and destroy heavily infected plants; don't leave them in the field.",
      "Stop overhead irrigation and keep leaves dry.",
      "Quickly ask an agronomist about a protective spray for nearby healthy plants.",
    ],
  },
  bacterial: {
    ar: [
      "أزل الأوراق المصابة، ولا تلمس النباتات وهي مبللة حتى لا تنقل العدوى.",
      "استخدم بذورًا وشتلات معتمدة وخالية من الأمراض.",
      "عقّم الأدوات بعد استخدامها، وطبّق دورة زراعية لمدة 2–3 سنوات.",
      "رشّ المركبات النحاسية قد يبطئ الانتشار، بعد استشارة مهندس زراعي.",
    ],
    en: [
      "Remove infected leaves and avoid handling plants while they are wet.",
      "Use certified, disease-free seeds and seedlings.",
      "Disinfect tools after use and rotate crops for 2–3 years.",
      "Copper-based sprays may slow the spread; check with an agronomist first.",
    ],
  },
  powderyMildew: {
    ar: [
      "حسّن التهوية وتعرّض النبات للشمس.",
      "أزل الأوراق المصابة بشدة.",
      "تجنّب الإفراط في التسميد النيتروجيني.",
      "رشّ الكبريت أو بيكربونات البوتاسيوم من الحلول الشائعة، بعد استشارة مختص.",
    ],
    en: [
      "Improve airflow and sun exposure.",
      "Remove heavily infected leaves.",
      "Avoid excess nitrogen fertilizer.",
      "Sulfur or potassium bicarbonate sprays are common options; check with a specialist.",
    ],
  },
  rust: {
    ar: [
      "أزل الأوراق المصابة مبكرًا لتقليل الجراثيم.",
      "ازرع أصنافًا مقاومة في المواسم القادمة.",
      "حسّن التهوية وتجنّب تبليل الأوراق.",
      "في الإصابات الشديدة، استشر مهندسًا زراعيًا بخصوص مبيد فطري مناسب.",
    ],
    en: [
      "Remove infected leaves early to reduce spores.",
      "Plant resistant varieties next season.",
      "Improve airflow and avoid wetting leaves.",
      "For severe cases, ask an agronomist about a suitable fungicide.",
    ],
  },
  rot: {
    ar: [
      "قلّم الأفرع المصابة وأزل الثمار المتحللة والجافة من على النبات ومن الأرض.",
      "أتلف المخلفات المصابة بعيدًا عن الحقل.",
      "عقّم أدوات التقليم بين كل نبات وآخر.",
      "الرش الوقائي في بداية الموسم يقلل الإصابة، بعد استشارة مهندس زراعي.",
    ],
    en: [
      "Prune infected branches and remove rotten or mummified fruit from the plant and ground.",
      "Destroy infected debris away from the field.",
      "Disinfect pruning tools between plants.",
      "Preventive sprays early in the season help; check with an agronomist.",
    ],
  },
  esca: {
    ar: [
      "لا يوجد علاج نهائي، والهدف تقليل الانتشار.",
      "قلّم الخشب المصاب في الجو الجاف وأتلفه.",
      "احمِ جروح التقليم بمادة مناسبة.",
      "استبدل الكروم شديدة الإصابة بعد استشارة مختص.",
    ],
    en: [
      "There's no cure; the goal is to limit the spread.",
      "Prune and destroy infected wood in dry weather.",
      "Protect pruning wounds with a suitable sealant.",
      "Replace severely infected vines after consulting a specialist.",
    ],
  },
  virusWhitefly: {
    ar: [
      "لا يوجد علاج للفيروس. اقلع النباتات المصابة لحماية الباقي.",
      "الفيروس ينتقل عن طريق الذبابة البيضاء: استخدم المصائد الصفراء اللاصقة والشباك.",
      "ازرع أصنافًا مقاومة، وتخلّص من الحشائش حول الحقل.",
    ],
    en: [
      "There's no cure for the virus. Remove infected plants to protect the rest.",
      "It's spread by whiteflies: use yellow sticky traps and insect nets.",
      "Plant resistant varieties and clear weeds around the field.",
    ],
  },
  virusContact: {
    ar: [
      "لا يوجد علاج للفيروس. اقلع النباتات المصابة.",
      "ينتقل باللمس والأدوات: اغسل يديك وعقّم الأدوات قبل لمس النباتات السليمة.",
      "استخدم بذورًا معتمدة وأصنافًا مقاومة.",
    ],
    en: [
      "There's no cure for the virus. Remove infected plants.",
      "It spreads by touch and tools: wash hands and disinfect tools before handling healthy plants.",
      "Use certified seeds and resistant varieties.",
    ],
  },
  greening: {
    ar: [
      "لا يوجد علاج. المرض ينتقل عن طريق حشرة بسيلا الموالح.",
      "كافح الحشرة الناقلة، وأزل الأشجار المصابة لحماية باقي البستان.",
      "اشترِ الشتلات من مشاتل معتمدة فقط.",
      "تواصل مع الإرشاد الزراعي، لأن المرض خطير على بساتين الموالح.",
    ],
    en: [
      "There's no cure. It's spread by the Asian citrus psyllid.",
      "Control the insect and remove infected trees to protect the orchard.",
      "Buy trees only from certified nurseries.",
      "Contact agricultural extension services; this disease is serious for citrus.",
    ],
  },
  mites: {
    ar: [
      "رشّ الماء على الوجه السفلي للأوراق لتقليل أعداد العنكبوت.",
      "زيت النيم أو الصابون البوتاسي من الحلول الآمنة نسبيًا.",
      "تجنّب المبيدات واسعة المدى التي تقتل الأعداء الطبيعيين للعنكبوت.",
      "الجفاف والحرارة يزيدان الإصابة، فحافظ على ري منتظم.",
    ],
    en: [
      "Spray water on leaf undersides to knock mites down.",
      "Neem oil or insecticidal soap are relatively safe options.",
      "Avoid broad-spectrum insecticides that kill the mites' natural predators.",
      "Heat and drought make it worse, so keep watering regular.",
    ],
  },
};

export const CLASS_ADVICE = {
  "Apple___Apple_scab": "fungal",
  "Apple___Black_rot": "rot",
  "Apple___Cedar_apple_rust": "rust",
  "Cherry_(including_sour)___Powdery_mildew": "powderyMildew",
  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "fungal",
  "Corn_(maize)___Common_rust_": "rust",
  "Corn_(maize)___Northern_Leaf_Blight": "fungal",
  "Grape___Black_rot": "rot",
  "Grape___Esca_(Black_Measles)": "esca",
  "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "fungal",
  "Orange___Haunglongbing_(Citrus_greening)": "greening",
  "Peach___Bacterial_spot": "bacterial",
  "Pepper,_bell___Bacterial_spot": "bacterial",
  "Potato___Early_blight": "fungal",
  "Potato___Late_blight": "lateBlight",
  "Squash___Powdery_mildew": "powderyMildew",
  "Strawberry___Leaf_scorch": "fungal",
  "Tomato___Bacterial_spot": "bacterial",
  "Tomato___Early_blight": "fungal",
  "Tomato___Late_blight": "lateBlight",
  "Tomato___Leaf_Mold": "fungal",
  "Tomato___Septoria_leaf_spot": "fungal",
  "Tomato___Spider_mites Two-spotted_spider_mite": "mites",
  "Tomato___Target_Spot": "fungal",
  "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "virusWhitefly",
  "Tomato___Tomato_mosaic_virus": "virusContact",
};

export function adviceFor(cls, lang) {
  const key = cls.healthy ? "healthy" : CLASS_ADVICE[cls.id] || "fungal";
  return ADVICE[key][lang];
}
