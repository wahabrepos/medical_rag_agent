# Patient-style questions with query rewrite, MedlinePlus and live PubMed

25 questions asked one by one through the running system (API, groq/openai/gpt-oss-120b, MiniCheck verifier), as free text with the model's own answer requested. Verdicts compare that answer with the expected diagnosis and were judged by hand.

- Evidence gate: **13 shown**, 12 withheld as "insufficient evidence"
- Model's answer vs expected: ✅ match 15, 🟡 partial 5, ❌ different 4, ⚪ declined 1

| # | Domain | Patient's words | Expected | Model's answer | Gate | Verdict |
|---|---|---|---|---|---|---|
| 1 | Cardiovascular | It feels like an elephant is sitting on my chest when I walk. | Stable coronary artery disease | Possible angina (chest pain on exertion) | withheld | ✅ match (angina, the symptom of stable coronary artery disease) |
| 2 | Cancer (Oncology) | I found a hard lump in my breast that doesn’t hurt and seems to be getting bigger. | Breast cancer (likely invasive carcinoma) | Diagnostic mammography is performed for people who have a lump or other signs or symptoms of breast cancer. | shown | ❌ different (says who gets diagnostic mammography, names no diagnosis) |
| 3 | Surgery (Acute Abdomen) | I have this sharp pain in my lower right belly that gets worse when I move. | Acute appendicitis | Appendicitis can cause sharp lower right abdominal pain that worsens over time. | shown | ✅ match (appendicitis; the shown statements only describe the pain) |
| 4 | Diabetes / Endocrinology | I’m always thirsty, peeing all the time, and losing weight without trying. | Diabetes mellitus (likely type 1 if acute) | Feeling very thirsty, urinating more often, and losing weight without trying are symptoms of diabetes. | shown | ✅ match (diabetes, from MedlinePlus symptom lists) |
| 5 | Infection | I’ve got a bad headache, stiff neck, and light really hurts my eyes. | Meningitis (likely bacterial until proven otherwise) | You should seek urgent medical evaluation as these symptoms may indicate a potentially serious condition such as meningitis. | shown | ✅ match (urgent evaluation for possible meningitis) |
| 6 | Ophthalmology | Suddenly I can’t see out of one eye, like a curtain came down. | Retinal detachment | A curtain‑like loss of vision in one eye is a symptom of retinal detachment. | shown | ✅ match (retinal detachment) |
| 7 | Orthopaedics | My hip hurts so much after I fell, and now I can’t stand on that leg. | Femoral neck fracture | You may have a hip fracture and should seek urgent medical evaluation. | withheld | 🟡 partial (hip fracture, not the femoral neck specifically) |
| 8 | Biochemistry / Metabolic | I feel really weak, my heart feels weird, and my muscles aren’t working right. | Hyperkalemia | Weakness developed within 3-5 days of illness and myalgia was present early in the disease course. | shown | ❌ different (restates a case report about weakness and myalgia) |
| 9 | Clinical Outcomes / Critical Care | I feel really sick, dizzy, and confused, and my family says I have a fever. | Sepsis / septic shock | Your symptoms could be consistent with sepsis. | shown | ✅ match (sepsis) |
| 10 | Cardiovascular (Emergency) | I suddenly felt a tearing pain in my chest that goes straight to my back. | Aortic dissection | Insufficient evidence | withheld | ⚪ declined (no diagnosis) |
| 11 | Cancer (Colorectal) | I’ve been seeing blood in my poop and my pants feel looser lately. | Colorectal cancer | Blood in the stool and looser bowel movements can be caused by several rectal conditions such as hemorrhoids, colorectal cancer, or other sources of lower gastrointestinal bleeding. | withheld | 🟡 partial (colorectal cancer listed among possible causes) |
| 12 | Cancer (Lung) | I keep coughing and sometimes there’s blood, and I’ve lost weight without trying. | Lung cancer | Your symptoms could be indicative of lung cancer. | withheld | ✅ match (lung cancer) |
| 13 | Surgery (Gallbladder) | I get this bad pain on the right side under my ribs after eating greasy food. | Cholelithiasis / cholecystitis | Greasy foods can trigger gallstone attacks | shown | ✅ match (gallstone attacks) |
| 14 | Surgery (Hernia) | There’s a bulge in my groin that pops out when I cough or lift things. | Inguinal hernia | You likely have an inguinal (groin) hernia. | withheld | ✅ match (inguinal hernia) |
| 15 | Diabetes / Endocrinology | My vision gets blurry sometimes and I feel tired all the time. | Type 2 diabetes mellitus | The symptoms may be indicative of diabetes. | withheld | 🟡 partial (diabetes, type not given) |
| 16 | Cardiovascular (Heart Failure) | I can’t lie flat at night because I feel like I’m drowning. | Congestive heart failure | Difficulty lying flat and shortness of breath when supine are symptoms of heart failure | shown | ✅ match (heart failure) |
| 17 | Cardiovascular (Arrhythmia) | My heart suddenly starts racing and feels like it’s skipping beats. | Atrial fibrillation | Your symptoms are consistent with an arrhythmia. | shown | 🟡 partial (arrhythmia, not atrial fibrillation) |
| 18 | Infection (Urinary) | It burns when I pee and I feel like I have to go all the time. | Urinary tract infection | You likely have a urinary tract infection. | shown | ✅ match (urinary tract infection) |
| 19 | Infection (Respiratory) | I’ve had a fever and chills, and my chest hurts when I breathe in. | Pneumonia | Your symptoms could be consistent with pneumonia. | withheld | ✅ match (pneumonia) |
| 20 | Ophthalmology | I see halos around lights and my eye hurts really badly. | Acute angle-closure glaucoma | Halos around lights are a symptom of cataracts; the eye pain is not explained by the provided information. | withheld | ❌ different (cataracts; the eye pain left unexplained) |
| 21 | Ophthalmology | Everything looks blurry in the center, but I can still see around it. | Age-related macular degeneration | You may have macular degeneration, which causes central vision blur while peripheral vision remains. | withheld | ✅ match (macular degeneration) |
| 22 | Orthopaedics | My knee suddenly got swollen and painful after I twisted it. | Anterior cruciate ligament (ACL) tear | You likely have an anterior cruciate ligament (ACL) injury. | withheld | ✅ match (ACL injury) |
| 23 | Orthopaedics | My back pain shoots down my leg like an electric shock. | Lumbar disc herniation | Your symptoms are consistent with sciatica. | shown | 🟡 partial (sciatica, not its cause (disc herniation)) |
| 24 | Biochemistry / Metabolic | My hands cramp up and my face feels twitchy. | Hypocalcemia | The symptoms are consistent with a peripheral nerve disorder. | shown | ❌ different (peripheral nerve disorder) |
| 25 | Clinical Outcomes / Shock | I feel faint, sweaty, and like I might pass out after losing a lot of blood. | Hypovolemic shock | You are likely experiencing a syncopal episode caused by low blood pressure from blood loss. | withheld | ✅ match (low blood pressure from blood loss) |

## Answers in full

### 1. Cardiovascular

> It feels like an elephant is sitting on my chest when I walk.

- Expected: Stable coronary artery disease (Angina pectoris)
- Shown to the user: **insufficient evidence**
- Model's own answer: Possible angina (chest pain on exertion)
- Evidence: partially_supported; 2 attempt(s); 9.9 s
- Verdict: ✅ match (angina, the symptom of stable coronary artery disease)

  - Chest pressure or squeezing pain can occur when the heart works harder, such as during walking (not found in the sources, support 0.02)
  - This type of chest pain is a symptom of coronary artery disease (source [3], support 0.96)
    > "Chest pain, pressure, tightness, or discomfort ... may spread to your neck, jaw, shoulder or arm."

  Retrieved sources:
  1. [Angina](https://medlineplus.gov/angina.html)
  2. [Chest Pain](https://medlineplus.gov/chestpain.html)
  3. [Coronary Artery Disease](https://medlineplus.gov/coronaryarterydisease.html)
  4. [Angina](https://medlineplus.gov/angina.html)
  5. [Pulmonary Embolism](https://medlineplus.gov/pulmonaryembolism.html)

### 2. Cancer (Oncology)

> I found a hard lump in my breast that doesn’t hurt and seems to be getting bigger.

- Expected: Breast cancer (likely invasive carcinoma) (Palpable breast mass with suspicious features)
- Shown to the user: **Diagnostic mammography is performed for people who have a lump or other signs or symptoms of breast cancer.**
- Model's own answer: Diagnostic mammography is performed for people who have a lump or other signs or symptoms of breast cancer.
- Evidence: supported; 2 attempt(s); 10.0 s
- Verdict: ❌ different (says who gets diagnostic mammography, names no diagnosis)

  - Diagnostic mammography is done for people who have a lump or other signs or symptoms of breast cancer. (source [1], support 0.95)
    > "It's done for people who have a lump or other signs or symptoms of breast cancer."

  Retrieved sources:
  1. [Mammography](https://medlineplus.gov/mammography.html)
  2. [Breast Diseases](https://medlineplus.gov/breastdiseases.html)
  3. [Breast Diseases](https://medlineplus.gov/breastdiseases.html)
  4. [Breast Cancer](https://medlineplus.gov/breastcancer.html)
  5. [Is the performance of MRI in preoperative staging of breast cancer independent of clinical and histological factors?](https://pubmed.ncbi.nlm.nih.gov/24836350/)

### 3. Surgery (Acute Abdomen)

> I have this sharp pain in my lower right belly that gets worse when I move.

- Expected: Acute appendicitis (Right lower quadrant tenderness with rebound)
- Shown to the user: **Appendicitis can cause sharp lower right abdominal pain that worsens over time.**
- Model's own answer: Appendicitis can cause sharp lower right abdominal pain that worsens over time.
- Evidence: supported; 2 attempt(s); 10.6 s
- Verdict: ✅ match (appendicitis; the shown statements only describe the pain)

  - pain can be sharp (source [1], support 0.99)
    > "It can be a sharp and stabbing pain in a specific spot, or a dull pain that is spread out."
  - pain gets worse over time (source [2], support 0.99)
    > "It is usually sudden and gets worse over time."

  Retrieved sources:
  1. [Pelvic Pain](https://medlineplus.gov/pelvicpain.html)
  2. [Appendicitis](https://medlineplus.gov/appendicitis.html)
  3. [Diverticulosis and Diverticulitis](https://medlineplus.gov/diverticulosisanddiverticulitis.html)
  4. [Diverticulosis and Diverticulitis](https://medlineplus.gov/diverticulosisanddiverticulitis.html)
  5. [Non-Drug Pain Management](https://medlineplus.gov/nondrugpainmanagement.html)

### 4. Diabetes / Endocrinology

> I’m always thirsty, peeing all the time, and losing weight without trying.

- Expected: Diabetes mellitus (likely type 1 if acute) (Polyuria, polydipsia, unexplained weight loss)
- Shown to the user: **Feeling very thirsty, urinating more often, and losing weight without trying are symptoms of diabetes.**
- Model's own answer: Feeling very thirsty, urinating more often, and losing weight without trying are symptoms of diabetes.
- Evidence: supported; 2 attempt(s); 10.6 s
- Verdict: ✅ match (diabetes, from MedlinePlus symptom lists)

  - Feeling very thirsty is a symptom of diabetes (source [1], support 0.97)
    > "Feeling very thirsty."
  - Urinating more often is a symptom of diabetes (source [1], support 0.98)
    > "Urinating (peeing) more often, including at night."
  - Losing weight without trying is a symptom of diabetes (source [1], support 0.97)
    > "Losing weight without trying."

  Retrieved sources:
  1. [Diabetes](https://medlineplus.gov/diabetes.html)
  2. [Diabetes Type 1](https://medlineplus.gov/diabetestype1.html)
  3. [Hyperglycemia](https://medlineplus.gov/hyperglycemia.html)
  4. [Diabetes Type 2](https://medlineplus.gov/diabetestype2.html)
  5. [Blood Glucose](https://medlineplus.gov/bloodglucose.html)

### 5. Infection

> I’ve got a bad headache, stiff neck, and light really hurts my eyes.

- Expected: Meningitis (likely bacterial until proven otherwise) (Neck stiffness, photophobia, meningismus)
- Shown to the user: **You should seek urgent medical evaluation as these symptoms may indicate a potentially serious condition such as meningitis.**
- Model's own answer: You should seek urgent medical evaluation as these symptoms may indicate a potentially serious condition such as meningitis.
- Evidence: supported; 1 attempt(s); 6.0 s
- Verdict: ✅ match (urgent evaluation for possible meningitis)

  - A headache with a stiff neck is listed as a warning sign that requires medical help (source [1], support 0.89)
    > "Get medical help right away if you have a headache after a blow to your head, or if you have a headache along with a stiff neck, fever, confusion, loss of consciousness, or pain in the eye or ear."
  - Neck stiffness is a typical warning sign of potentially life‑threatening headache disorders (source [4], support 0.98)
    > "A patient is suspected of suffering from a potentially life threatening headache according to the following typical warning signs: sudden onset headache, worst headache, neurological findings, fever, neck stiffness, epileptic seizures and deterioration of vigilance."

  Retrieved sources:
  1. [Headache](https://medlineplus.gov/headache.html)
  2. [Migraine](https://medlineplus.gov/migraine.html)
  3. [Headache](https://medlineplus.gov/headache.html)
  4. [[Symptomatic headache. Essential differential diagnosis].](https://pubmed.ncbi.nlm.nih.gov/15316763/)
  5. [Pseudomigraine with lymphocytic pleocytosis: a calcium channelopathy?](https://pubmed.ncbi.nlm.nih.gov/12940811/)

### 6. Ophthalmology

> Suddenly I can’t see out of one eye, like a curtain came down.

- Expected: Retinal detachment (Acute monocular vision loss (amaurosis/retinal detachment description))
- Shown to the user: **A curtain‑like loss of vision in one eye is a symptom of retinal detachment.**
- Model's own answer: A curtain‑like loss of vision in one eye is a symptom of retinal detachment.
- Evidence: supported; 2 attempt(s); 12.0 s
- Verdict: ✅ match (retinal detachment)

  - Curtain‑like visual field defect is a presentation of retinal detachment (source [2], support 0.99)
    > "Retinal detachment presents with a curtain-like visual field defect and usually requires surgical intervention."
  - Patients may describe a curtain over the visual field in retinal detachment (source [1], support 0.95)
    > "It may also seem like there is a "curtain" over your field of vision."

  Retrieved sources:
  1. [Retinal Detachment](https://medlineplus.gov/retinaldetachment.html)
  2. [Sudden Vision Loss: A Diagnostic Approach.](https://pubmed.ncbi.nlm.nih.gov/39823616/)
  3. [Optic Nerve Disorders](https://medlineplus.gov/opticnervedisorders.html)
  4. [Retinal Disorders](https://medlineplus.gov/retinaldisorders.html)
  5. [Retinal Disorders](https://medlineplus.gov/retinaldisorders.html)

### 7. Orthopaedics

> My hip hurts so much after I fell, and now I can’t stand on that leg.

- Expected: Femoral neck fracture (Inability to bear weight with suspected hip fracture)
- Shown to the user: **insufficient evidence**
- Model's own answer: You may have a hip fracture and should seek urgent medical evaluation.
- Evidence: partially_supported; 2 attempt(s); 9.9 s
- Verdict: 🟡 partial (hip fracture, not the femoral neck specifically)

  - Intense hip pain after a fall and inability to stand are symptoms of a fracture (not found in the sources, support 0.02)
  - Falls can cause hip fractures (source [2], support 0.98)
    > "However, playing sports, running, overuse, or falling can sometimes lead to hip injuries such as: ... Fractures (broken bones)."

  Retrieved sources:
  1. [Hip Injuries and Disorders](https://medlineplus.gov/hipinjuriesanddisorders.html)
  2. [Hip Injuries and Disorders](https://medlineplus.gov/hipinjuriesanddisorders.html)
  3. [Hip Replacement](https://medlineplus.gov/hipreplacement.html)
  4. [Fractures](https://medlineplus.gov/fractures.html)
  5. [Hip Replacement](https://medlineplus.gov/hipreplacement.html)

### 8. Biochemistry / Metabolic

> I feel really weak, my heart feels weird, and my muscles aren’t working right.

- Expected: Hyperkalemia (Muscle weakness with cardiac arrhythmia symptoms)
- Shown to the user: **Weakness developed within 3-5 days of illness and myalgia was present early in the disease course.**
- Model's own answer: Weakness developed within 3-5 days of illness and myalgia was present early in the disease course.
- Evidence: supported; 2 attempt(s); 10.1 s
- Verdict: ❌ different (restates a case report about weakness and myalgia)

  - Weakness developed within 3-5 days of illness (source [1], support 0.99)
    > "Weakness developed within 3-5 days of illness, which was severe in 4 and moderate in 3 patients."
  - Myalgia was present (source [1], support 0.99)
    > "Fever was present in all and myalgia in 5 patients."

  Retrieved sources:
  1. [Acute pure motor quadriplegia: is it dengue myositis?](https://pubmed.ncbi.nlm.nih.gov/16315973/)
  2. [Myasthenia Gravis](https://medlineplus.gov/myastheniagravis.html)
  3. [Hypothyroidism](https://medlineplus.gov/hypothyroidism.html)
  4. [Do traditional symptoms of hypothyroidism correlate with biochemical disease?](https://pubmed.ncbi.nlm.nih.gov/9294788/)
  5. [Heart Failure](https://medlineplus.gov/heartfailure.html)

### 9. Clinical Outcomes / Critical Care

> I feel really sick, dizzy, and confused, and my family says I have a fever.

- Expected: Sepsis / septic shock (Altered mental status with fever and hypotension)
- Shown to the user: **Your symptoms could be consistent with sepsis.**
- Model's own answer: Your symptoms could be consistent with sepsis.
- Evidence: supported; 1 attempt(s); 6.1 s
- Verdict: ✅ match (sepsis)

  - Confusion or disorientation is a symptom of sepsis (source [3], support 0.94)
    > "Confusion or disorientation."
  - Fever is a symptom of sepsis (source [3], support 0.98)
    > "Fever, shivering, or feeling very cold."

  Retrieved sources:
  1. [Dizziness and Vertigo](https://medlineplus.gov/dizzinessandvertigo.html)
  2. [Anemia](https://medlineplus.gov/anemia.html)
  3. [Sepsis](https://medlineplus.gov/sepsis.html)
  4. [Encephalitis](https://medlineplus.gov/encephalitis.html)
  5. [Encephalitis](https://medlineplus.gov/encephalitis.html)

### 10. Cardiovascular (Emergency)

> I suddenly felt a tearing pain in my chest that goes straight to my back.

- Expected: Aortic dissection (Tearing chest pain radiating to the back)
- Shown to the user: **insufficient evidence**
- Model's own answer: Insufficient evidence
- Evidence: not_supported; 2 attempt(s); 11.1 s
- Verdict: ⚪ declined (no diagnosis)

  - The provided passages do not mention tearing chest pain radiating to the back as a symptom of any condition. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Chest Pain](https://medlineplus.gov/chestpain.html)
  2. [Heart Attack](https://medlineplus.gov/heartattack.html)
  3. [Chest Pain](https://medlineplus.gov/chestpain.html)
  4. [Does esophageal function vary at the striated and smooth muscle segments in functional chest pain?](https://pubmed.ncbi.nlm.nih.gov/12358233/)
  5. [Heart Disease in Women](https://medlineplus.gov/heartdiseaseinwomen.html)

### 11. Cancer (Colorectal)

> I’ve been seeing blood in my poop and my pants feel looser lately.

- Expected: Colorectal cancer (Hematochezia with unintentional weight loss)
- Shown to the user: **insufficient evidence**
- Model's own answer: Blood in the stool and looser bowel movements can be caused by several rectal conditions such as hemorrhoids, colorectal cancer, or other sources of lower gastrointestinal bleeding.
- Evidence: supported; 1 attempt(s); 7.9 s
- Verdict: 🟡 partial (colorectal cancer listed among possible causes)

  - Bright red or very dark blood in stool is a symptom of colorectal cancer (source [2], support 0.66)
    > "Blood (either bright red or very dark) in the stool."
  - Bright red blood in stool is a symptom of hemorrhoids (source [3], support 0.98)
    > "you would see bright red blood in your stool, on toilet paper, or in the toilet bowl after a bowel movement."
  - Blood in the lower digestive tract is a sign of gastrointestinal bleeding (source [1], support 0.81)
    > "Dark blood mixed with stool."
  - A change in bowel habits such as looser stools is a symptom of colorectal cancer (not found in the sources, support 0.48)

  Retrieved sources:
  1. [Gastrointestinal Bleeding](https://medlineplus.gov/gastrointestinalbleeding.html)
  2. [Colorectal Cancer](https://medlineplus.gov/colorectalcancer.html)
  3. [Hemorrhoids](https://medlineplus.gov/hemorrhoids.html)
  4. [Rectal Disorders](https://medlineplus.gov/rectaldisorders.html)
  5. [Is pelvic pain associated with defecatory symptoms in women with pelvic organ prolapse?](https://pubmed.ncbi.nlm.nih.gov/21394762/)

### 12. Cancer (Lung)

> I keep coughing and sometimes there’s blood, and I’ve lost weight without trying.

- Expected: Lung cancer (Chronic cough with hemoptysis and cachexia)
- Shown to the user: **insufficient evidence**
- Model's own answer: Your symptoms could be indicative of lung cancer.
- Evidence: partially_supported; 2 attempt(s); 10.2 s
- Verdict: ✅ match (lung cancer)

  - Cough that doesn't go away or gets worse over time can be a symptom of lung cancer (source [5], support 0.97)
    > "A cough that doesn't go away or gets worse over time."
  - Coughing up blood can be a symptom of lung cancer (not found in the sources, support 0.00)
  - Weight loss for no known reason can be a symptom of lung cancer (source [5], support 0.91)
    > "Weight loss for no known reason."

  Retrieved sources:
  1. [Cough](https://medlineplus.gov/cough.html)
  2. [Chronic Bronchitis](https://medlineplus.gov/chronicbronchitis.html)
  3. [Mycobacterial Infections](https://medlineplus.gov/mycobacterialinfections.html)
  4. [Body Weight](https://medlineplus.gov/bodyweight.html)
  5. [Lung Cancer](https://medlineplus.gov/lungcancer.html)

### 13. Surgery (Gallbladder)

> I get this bad pain on the right side under my ribs after eating greasy food.

- Expected: Cholelithiasis / cholecystitis (Right upper quadrant pain postprandial (biliary colic))
- Shown to the user: **Greasy foods can trigger gallstone attacks**
- Model's own answer: Greasy foods can trigger gallstone attacks
- Evidence: supported; 2 attempt(s); 10.3 s
- Verdict: ✅ match (gallstone attacks)

  - Gallstone attacks usually happen after you eat (source [1], support 0.99)
    > "Gallstone attacks usually happen after you eat."

  Retrieved sources:
  1. [Gallstones](https://medlineplus.gov/gallstones.html)
  2. [Gallbladder Cancer](https://medlineplus.gov/gallbladdercancer.html)
  3. [Gallbladder Diseases](https://medlineplus.gov/gallbladderdiseases.html)
  4. [The clinical significance of bile duct sludge: is it different from bile duct stones?](https://pubmed.ncbi.nlm.nih.gov/17342562/)
  5. [Does endoscopic ultrasound have anything to offer in the diagnosis of idiopathic acute pancreatitis?](https://pubmed.ncbi.nlm.nih.gov/19287106/)

### 14. Surgery (Hernia)

> There’s a bulge in my groin that pops out when I cough or lift things.

- Expected: Inguinal hernia (Reducible inguinal mass with Valsalva)
- Shown to the user: **insufficient evidence**
- Model's own answer: You likely have an inguinal (groin) hernia.
- Evidence: not_supported; 2 attempt(s); 8.9 s
- Verdict: ✅ match (inguinal hernia)

  - A bulge in the groin that appears when coughing or lifting is characteristic of an inguinal hernia (not found in the sources, support 0.01)

  Retrieved sources:
  1. [Hernia](https://medlineplus.gov/hernia.html)
  2. [Hernia](https://medlineplus.gov/hernia.html)
  3. [Are voiding symptoms really associated with abnormal urodynamic voiding parameters in women?](https://pubmed.ncbi.nlm.nih.gov/19054175/)
  4. [Hiatal Hernia](https://medlineplus.gov/hiatalhernia.html)
  5. [Is endoscopic totally extraperitoneal hernioplasty justified for the repair of groin hernia in female patients?](https://pubmed.ncbi.nlm.nih.gov/16308798/)

### 15. Diabetes / Endocrinology

> My vision gets blurry sometimes and I feel tired all the time.

- Expected: Type 2 diabetes mellitus (Intermittent blurred vision with fatigue (hyperglycemia-related))
- Shown to the user: **insufficient evidence**
- Model's own answer: The symptoms may be indicative of diabetes.
- Evidence: not_supported; 2 attempt(s); 10.4 s
- Verdict: 🟡 partial (diabetes, type not given)

  - Blurry vision is a symptom of diabetes (not found in the sources, support 0.00)
  - Fatigue is a symptom of diabetes (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Retinal Disorders](https://medlineplus.gov/retinaldisorders.html)
  2. [Cataract](https://medlineplus.gov/cataract.html)
  3. [Diabetes](https://medlineplus.gov/diabetes.html)
  4. [Retinal Disorders](https://medlineplus.gov/retinaldisorders.html)
  5. [Myasthenia Gravis](https://medlineplus.gov/myastheniagravis.html)

### 16. Cardiovascular (Heart Failure)

> I can’t lie flat at night because I feel like I’m drowning.

- Expected: Congestive heart failure (Orthopnea)
- Shown to the user: **Difficulty lying flat and shortness of breath when supine are symptoms of heart failure**
- Model's own answer: Difficulty lying flat and shortness of breath when supine are symptoms of heart failure
- Evidence: supported; 2 attempt(s); 10.9 s
- Verdict: ✅ match (heart failure)

  - Difficulty sleeping when lying flat is a symptom of heart failure (source [1], support 0.98)
    > "Difficulty sleeping when lying flat."
  - Feeling short of breath is a symptom of heart failure (source [1], support 0.98)
    > "Feeling short of breath (like you can't get enough air) when you do things like climbing stairs."

  Retrieved sources:
  1. [Heart Failure](https://medlineplus.gov/heartfailure.html)
  2. [Breathing Problems](https://medlineplus.gov/breathingproblems.html)
  3. [Collapsed Lung](https://medlineplus.gov/collapsedlung.html)
  4. [Are voiding symptoms really associated with abnormal urodynamic voiding parameters in women?](https://pubmed.ncbi.nlm.nih.gov/19054175/)
  5. [Is there a relationship between preferred posture and positioning in early life and the direction of subsequent asymmetrical postural deformity in non ambulant people with cerebral palsy?](https://pubmed.ncbi.nlm.nih.gov/18796054/)

### 17. Cardiovascular (Arrhythmia)

> My heart suddenly starts racing and feels like it’s skipping beats.

- Expected: Atrial fibrillation (Palpitations with irregular rhythm)
- Shown to the user: **Your symptoms are consistent with an arrhythmia.**
- Model's own answer: Your symptoms are consistent with an arrhythmia.
- Evidence: supported; 1 attempt(s); 6.3 s
- Verdict: 🟡 partial (arrhythmia, not atrial fibrillation)

  - Fast or slow heartbeat is a symptom of an arrhythmia (source [1], support 0.98)
    > "Fast or slow heartbeat."
  - Skipping, fluttering, or pounding heartbeats is a symptom of an arrhythmia (source [1], support 0.97)
    > "Skipping, fluttering, or pounding heartbeats."

  Retrieved sources:
  1. [Arrhythmia](https://medlineplus.gov/arrhythmia.html)
  2. [Arrhythmia](https://medlineplus.gov/arrhythmia.html)
  3. [Arrhythmia](https://medlineplus.gov/arrhythmia.html)
  4. [Do ventricular arrhythmias in athletes subside over time?](https://pubmed.ncbi.nlm.nih.gov/24340838/)
  5. [Atrial Fibrillation](https://medlineplus.gov/atrialfibrillation.html)

### 18. Infection (Urinary)

> It burns when I pee and I feel like I have to go all the time.

- Expected: Urinary tract infection (Dysuria with urinary frequency)
- Shown to the user: **You likely have a urinary tract infection.**
- Model's own answer: You likely have a urinary tract infection.
- Evidence: supported; 1 attempt(s); 5.6 s
- Verdict: ✅ match (urinary tract infection)

  - Burning pain when urinating is a symptom of a urinary tract infection (source [1], support 0.93)
    > "Pain or burning when you urinate."
  - Feeling the need to urinate often is a symptom of a urinary tract infection (source [1], support 0.97)
    > "An urge to urinate often."

  Retrieved sources:
  1. [Urinary Tract Infections](https://medlineplus.gov/urinarytractinfections.html)
  2. [Urinary Incontinence](https://medlineplus.gov/urinaryincontinence.html)
  3. [Interstitial Cystitis](https://medlineplus.gov/interstitialcystitis.html)
  4. [Does treatment with clean intermittent catheterization in boys with posterior urethral valves affect bladder and renal function?](https://pubmed.ncbi.nlm.nih.gov/14501691/)
  5. [Prostate Diseases](https://medlineplus.gov/prostatediseases.html)

### 19. Infection (Respiratory)

> I’ve had a fever and chills, and my chest hurts when I breathe in.

- Expected: Pneumonia (Fever with pleuritic chest pain)
- Shown to the user: **insufficient evidence**
- Model's own answer: Your symptoms could be consistent with pneumonia.
- Evidence: partially_supported; 2 attempt(s); 9.3 s
- Verdict: ✅ match (pneumonia)

  - Fever and chills are symptoms of pneumonia (not found in the sources, support 0.00)
  - Chest pain when you breathe in is a symptom of pneumonia (source [3], support 0.94)
    > "Chest pain when you breathe or cough."

  Retrieved sources:
  1. [Chest Pain](https://medlineplus.gov/chestpain.html)
  2. [Chest Pain](https://medlineplus.gov/chestpain.html)
  3. [Pneumonia](https://medlineplus.gov/pneumonia.html)
  4. [Is there an anatomic explanation for chest pain in patients with pulmonary sarcoidosis?](https://pubmed.ncbi.nlm.nih.gov/9305302/)
  5. [Endocarditis](https://medlineplus.gov/endocarditis.html)

### 20. Ophthalmology

> I see halos around lights and my eye hurts really badly.

- Expected: Acute angle-closure glaucoma (Halos with ocular pain and increased intraocular pressure)
- Shown to the user: **insufficient evidence**
- Model's own answer: Halos around lights are a symptom of cataracts; the eye pain is not explained by the provided information.
- Evidence: not_supported; 2 attempt(s); 9.3 s
- Verdict: ❌ different (cataracts; the eye pain left unexplained)

  - Halos around lights are a symptom of cataracts (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Eye Infections](https://medlineplus.gov/eyeinfections.html)
  2. [Refractive Errors](https://medlineplus.gov/refractiveerrors.html)
  3. [Diabetic Eye Problems](https://medlineplus.gov/diabeticeyeproblems.html)
  4. [Cataract](https://medlineplus.gov/cataract.html)
  5. [Are all aberrations equal?](https://pubmed.ncbi.nlm.nih.gov/12361157/)

### 21. Ophthalmology

> Everything looks blurry in the center, but I can still see around it.

- Expected: Age-related macular degeneration (Central vision loss (macular involvement))
- Shown to the user: **insufficient evidence**
- Model's own answer: You may have macular degeneration, which causes central vision blur while peripheral vision remains.
- Evidence: partially_supported; 2 attempt(s); 8.8 s
- Verdict: ✅ match (macular degeneration)

  - Blurry or hazy central vision with preserved peripheral vision is described as central vision loss (not found in the sources, support 0.10)
  - Macular degeneration causes loss of central vision (source [2], support 0.98)
    > "It is a disease that destroys your sharp, central vision."

  Retrieved sources:
  1. [Amblyopia: is visual loss permanent?](https://pubmed.ncbi.nlm.nih.gov/10966943/)
  2. [Macular Degeneration](https://medlineplus.gov/maculardegeneration.html)
  3. [Retinal Disorders](https://medlineplus.gov/retinaldisorders.html)
  4. [Vision Impairment and Blindness](https://medlineplus.gov/visionimpairmentandblindness.html)
  5. [Retinal Disorders](https://medlineplus.gov/retinaldisorders.html)

### 22. Orthopaedics

> My knee suddenly got swollen and painful after I twisted it.

- Expected: Anterior cruciate ligament (ACL) tear (Acute knee effusion after rotational injury)
- Shown to the user: **insufficient evidence**
- Model's own answer: You likely have an anterior cruciate ligament (ACL) injury.
- Evidence: partially_supported; 2 attempt(s); 7.4 s
- Verdict: ✅ match (ACL injury)

  - A sudden twisting motion can injure the ACL (source [3], support 0.98)
    > "The classic presentation is a non-contact twisting injury with an audible pop and the rapid onset of swelling."
  - ACL injury often presents with rapid onset swelling (not found in the sources, support 0.12)

  Retrieved sources:
  1. [Joint Disorders](https://medlineplus.gov/jointdisorders.html)
  2. [Sprains and Strains](https://medlineplus.gov/sprainsandstrains.html)
  3. [Anterior cruciate ligament tears for the primary care sports physician: what to know on the field and in the office.](https://pubmed.ncbi.nlm.nih.gov/26559706/)
  4. [Knee Injuries and Disorders](https://medlineplus.gov/kneeinjuriesanddisorders.html)
  5. [Is intra-articular pathology associated with MCL edema on MR imaging of the non-traumatic knee?](https://pubmed.ncbi.nlm.nih.gov/15940487/)

### 23. Orthopaedics

> My back pain shoots down my leg like an electric shock.

- Expected: Lumbar disc herniation (Radicular pain (sciatica))
- Shown to the user: **Your symptoms are consistent with sciatica.**
- Model's own answer: Your symptoms are consistent with sciatica.
- Evidence: supported; 1 attempt(s); 9.6 s
- Verdict: 🟡 partial (sciatica, not its cause (disc herniation))

  - Pain that starts in the lower back and extends down the leg is a symptom of sciatica (source [1], support 0.90)
    > "It can start in the lower back and extend down your leg to your calf, foot, or even your toes."

  Retrieved sources:
  1. [Sciatica](https://medlineplus.gov/sciatica.html)
  2. [Recommendations for terminology and the identification of neuropathic pain in people with spine-related leg pain. Outcomes from the NeuPSIG working group.](https://pubmed.ncbi.nlm.nih.gov/37235637/)
  3. [Electrical Injuries](https://medlineplus.gov/electricalinjuries.html)
  4. [Restless Legs](https://medlineplus.gov/restlesslegs.html)
  5. [Do age and comorbidity affect quality of life or PTA-induced quality-of-life improvements in patients with symptomatic pad?](https://pubmed.ncbi.nlm.nih.gov/15943516/)

### 24. Biochemistry / Metabolic

> My hands cramp up and my face feels twitchy.

- Expected: Hypocalcemia (Tetany (neuromuscular irritability))
- Shown to the user: **The symptoms are consistent with a peripheral nerve disorder.**
- Model's own answer: The symptoms are consistent with a peripheral nerve disorder.
- Evidence: supported; 1 attempt(s); 5.2 s
- Verdict: ❌ different (peripheral nerve disorder)

  - Muscle cramps or twitching are symptoms of peripheral nerve disorders (source [3], support 0.77)
    > "Cramps or twitching muscles."

  Retrieved sources:
  1. [Bell's Palsy](https://medlineplus.gov/bellspalsy.html)
  2. [Dystonia](https://medlineplus.gov/dystonia.html)
  3. [Peripheral Nerve Disorders](https://medlineplus.gov/peripheralnervedisorders.html)
  4. [Orthostatic myoclonus: an underrecognized cause of unsteadiness?](https://pubmed.ncbi.nlm.nih.gov/23916653/)
  5. [Neuromuscular Disorders](https://medlineplus.gov/neuromusculardisorders.html)

### 25. Clinical Outcomes / Shock

> I feel faint, sweaty, and like I might pass out after losing a lot of blood.

- Expected: Hypovolemic shock (Hypotension with signs of hypoperfusion)
- Shown to the user: **insufficient evidence**
- Model's own answer: You are likely experiencing a syncopal episode caused by low blood pressure from blood loss.
- Evidence: partially_supported; 2 attempt(s); 11.1 s
- Verdict: ✅ match (low blood pressure from blood loss)

  - Fainting occurs when blood pressure drops suddenly, causing decreased brain blood flow (source [1], support 0.98)
    > "Fainting usually happens when your blood pressure drops suddenly, causing a decrease in blood flow to your brain."
  - Hypotensive events are significantly associated with symptomatic episodes (not found in the sources, support 0.13)

  Retrieved sources:
  1. [Fainting](https://medlineplus.gov/fainting.html)
  2. [Anemia](https://medlineplus.gov/anemia.html)
  3. [Hypoglycemia](https://medlineplus.gov/hypoglycemia.html)
  4. [Hypotension in patients with coronary disease: can profound hypotensive events cause myocardial ischaemic events?](https://pubmed.ncbi.nlm.nih.gov/10490564/)
  5. [Are voiding symptoms really associated with abnormal urodynamic voiding parameters in women?](https://pubmed.ncbi.nlm.nih.gov/19054175/)
