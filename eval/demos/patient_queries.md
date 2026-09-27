# Patient-style questions: the semantic gap

25 questions asked one by one through the running system (API, groq/openai/gpt-oss-120b, MiniCheck verifier), as free text with the model's own answer requested. Verdicts compare that answer with the expected diagnosis and were judged by hand.

- Evidence gate: **2 shown**, 23 withheld as "insufficient evidence"
- Model's answer vs expected: ✅ match 7, 🟡 partial 3, ❌ different 3, ⚪ declined 12

| # | Domain | Patient's words | Expected | Model's answer | Gate | Verdict |
|---|---|---|---|---|---|---|
| 1 | Cardiovascular | It feels like an elephant is sitting on my chest when I walk. | Stable coronary artery disease | Insufficient evidence | withheld | ⚪ declined (no diagnosis; retrieval found the 'elephant trunk' surgical technique) |
| 2 | Cancer (Oncology) | I found a hard lump in my breast that doesn’t hurt and seems to be getting bigger. | Breast cancer (likely invasive carcinoma) | Palpable breast masses are typically assessed with ultrasound followed by core needle biopsy to establish a diagnosis. | shown | ❌ different (describes the work-up (ultrasound, core biopsy), names no diagnosis) |
| 3 | Surgery (Acute Abdomen) | I have this sharp pain in my lower right belly that gets worse when I move. | Acute appendicitis | Insufficient evidence | withheld | ⚪ declined (no diagnosis) |
| 4 | Diabetes / Endocrinology | I’m always thirsty, peeing all the time, and losing weight without trying. | Diabetes mellitus (likely type 1 if acute) | Your symptoms may suggest uncontrolled diabetes and you should seek medical evaluation promptly. | withheld | ✅ match (uncontrolled diabetes) |
| 5 | Infection | I’ve got a bad headache, stiff neck, and light really hurts my eyes. | Meningitis (likely bacterial until proven otherwise) | Your symptoms could be concerning for meningitis, which typically presents with headache, neck stiffness, and photophobia. | withheld | ✅ match (meningitis) |
| 6 | Ophthalmology | Suddenly I can’t see out of one eye, like a curtain came down. | Retinal detachment | Sudden vision loss in one eye described as a curtain coming down suggests retinal detachment and requires urgent ophthalmologic evaluation. | withheld | ✅ match (retinal detachment) |
| 7 | Orthopaedics | My hip hurts so much after I fell, and now I can’t stand on that leg. | Femoral neck fracture | You should have your hip evaluated promptly, including imaging (e.g., X‑ray) and orthopedic assessment, as severe pain and inability to bear weight after a fall may indicate a fracture or other serious injury. | withheld | 🟡 partial (possible fracture, not specifically femoral neck) |
| 8 | Biochemistry / Metabolic | I feel really weak, my heart feels weird, and my muscles aren’t working right. | Hyperkalemia | Insufficient evidence | withheld | ⚪ declined (no diagnosis) |
| 9 | Clinical Outcomes / Critical Care | I feel really sick, dizzy, and confused, and my family says I have a fever. | Sepsis / septic shock | Fever is a defining feature of influenza‑like illness and systemic symptoms such as dizziness or confusion may occur, but clinical evaluation is required to determine the cause and appropriate treatment. | withheld | ❌ different (influenza-like illness, not sepsis) |
| 10 | Cardiovascular (Emergency) | I suddenly felt a tearing pain in my chest that goes straight to my back. | Aortic dissection | insufficient evidence | withheld | ⚪ declined (no diagnosis) |
| 11 | Cancer (Colorectal) | I’ve been seeing blood in my poop and my pants feel looser lately. | Colorectal cancer | Changes in bowel habits such as looser stools are recognized as symptoms that raise suspicion for colorectal cancer and other serious conditions. | shown | 🟡 partial (names colorectal cancer, but reads 'pants feel looser' as looser stools and ignores the blood) |
| 12 | Cancer (Lung) | I keep coughing and sometimes there’s blood, and I’ve lost weight without trying. | Lung cancer | Insufficient evidence | withheld | ⚪ declined (no diagnosis) |
| 13 | Surgery (Gallbladder) | I get this bad pain on the right side under my ribs after eating greasy food. | Cholelithiasis / cholecystitis | Your symptoms may suggest biliary colic or gallbladder disease, which can cause right-sided upper abdominal pain after fatty meals. | withheld | ✅ match (biliary colic / gallbladder disease) |
| 14 | Surgery (Hernia) | There’s a bulge in my groin that pops out when I cough or lift things. | Inguinal hernia | insufficient evidence | withheld | ⚪ declined (no diagnosis) |
| 15 | Diabetes / Endocrinology | My vision gets blurry sometimes and I feel tired all the time. | Type 2 diabetes mellitus | Insufficient evidence to determine the cause of blurry vision and fatigue. | withheld | ⚪ declined (no diagnosis) |
| 16 | Cardiovascular (Heart Failure) | I can’t lie flat at night because I feel like I’m drowning. | Congestive heart failure | insufficient evidence | withheld | ⚪ declined (no diagnosis) |
| 17 | Cardiovascular (Arrhythmia) | My heart suddenly starts racing and feels like it’s skipping beats. | Atrial fibrillation | Ventricular premature beats and other ventricular arrhythmias can cause a rapid, irregular heart sensation. | withheld | ❌ different (ventricular arrhythmias, not atrial fibrillation) |
| 18 | Infection (Urinary) | It burns when I pee and I feel like I have to go all the time. | Urinary tract infection | The provided passages do not contain information about burning during urination or a frequent urge to urinate. | withheld | ⚪ declined (no diagnosis) |
| 19 | Infection (Respiratory) | I’ve had a fever and chills, and my chest hurts when I breathe in. | Pneumonia | Your symptoms of fever, chills, and chest pain on breathing may indicate a respiratory infection such as pneumonia and should be evaluated promptly, especially if you have neutropenia. | withheld | ✅ match (pneumonia) |
| 20 | Ophthalmology | I see halos around lights and my eye hurts really badly. | Acute angle-closure glaucoma | Insufficient evidence | withheld | ⚪ declined (no diagnosis) |
| 21 | Ophthalmology | Everything looks blurry in the center, but I can still see around it. | Age-related macular degeneration | insufficient evidence | withheld | ⚪ declined (no diagnosis) |
| 22 | Orthopaedics | My knee suddenly got swollen and painful after I twisted it. | Anterior cruciate ligament (ACL) tear | You may have an acute ligament injury such as an ACL tear, which often causes sudden swelling and pain after a twisting motion. | withheld | ✅ match (ACL tear) |
| 23 | Orthopaedics | My back pain shoots down my leg like an electric shock. | Lumbar disc herniation | Your description is characteristic of sciatica, which is often caused by lumbar nerve root irritation. | withheld | 🟡 partial (sciatica from nerve root irritation; does not name disc herniation) |
| 24 | Biochemistry / Metabolic | My hands cramp up and my face feels twitchy. | Hypocalcemia | Insufficient evidence | withheld | ⚪ declined (no diagnosis) |
| 25 | Clinical Outcomes / Shock | I feel faint, sweaty, and like I might pass out after losing a lot of blood. | Hypovolemic shock | Your symptoms are likely due to low blood pressure from significant blood loss (hypovolemia). | withheld | ✅ match (hypovolemia from blood loss) |

## Answers in full

### 1. Cardiovascular

> It feels like an elephant is sitting on my chest when I walk.

- Expected: Stable coronary artery disease (Angina pectoris)
- Shown to the user: **insufficient evidence**
- Model's own answer: Insufficient evidence
- Evidence: not_supported; 2 attempt(s); 3.6 s
- Verdict: ⚪ declined (no diagnosis; retrieval found the 'elephant trunk' surgical technique)

  - The provided passages discuss complications of the elephant trunk technique and unrelated topics; none address chest discomfort on walking. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Elephant trunk in a small-calibre true lumen for chronic aortic dissection: cause of haemolytic anaemia?](https://pubmed.ncbi.nlm.nih.gov/25228241/)
  2. [Elephant trunk in a small-calibre true lumen for chronic aortic dissection: cause of haemolytic anaemia?](https://pubmed.ncbi.nlm.nih.gov/25228241/)
  3. [Is there an anatomic explanation for chest pain in patients with pulmonary sarcoidosis?](https://pubmed.ncbi.nlm.nih.gov/9305302/)
  4. [Are Changes in Gait and Balance Across the Disease Step Rating Scale in Multiple Sclerosis Statistically Significant and Clinically Meaningful?](https://pubmed.ncbi.nlm.nih.gov/27109334/)
  5. [Does Degenerative Lumbar Spine Disease Influence Femoroacetabular Flexion in Patients Undergoing Total Hip Arthroplasty?](https://pubmed.ncbi.nlm.nih.gov/27020429/)

### 2. Cancer (Oncology)

> I found a hard lump in my breast that doesn’t hurt and seems to be getting bigger.

- Expected: Breast cancer (likely invasive carcinoma) (Palpable breast mass with suspicious features)
- Shown to the user: **Palpable breast masses are typically assessed with ultrasound followed by core needle biopsy to establish a diagnosis.**
- Model's own answer: Palpable breast masses are typically assessed with ultrasound followed by core needle biopsy to establish a diagnosis.
- Evidence: supported; 2 attempt(s); 2.9 s
- Verdict: ❌ different (describes the work-up (ultrasound, core biopsy), names no diagnosis)

  - Palpable breast lesions are evaluated with ultrasound and needle core biopsy (source [1], support 0.84)
    > "All underwent ultrasound, SWE and needle core biopsy."
  - Core needle biopsy is performed on breast lesions identified on imaging (source [2], support 0.63)
    > "A total of 360 core needle biopsies was obtained from 169 breast lesions in 146 patients."

  Retrieved sources:
  1. [Shear-wave elastography and greyscale assessment of palpable probably benign masses: is biopsy always required?](https://pubmed.ncbi.nlm.nih.gov/27007593/)
  2. [Three-dimensional ultrasound-validated large-core needle biopsy: is it a reliable method for the histological assessment of breast lesions?](https://pubmed.ncbi.nlm.nih.gov/15065192/)
  3. [Palpable breast masses with probably benign morphology at sonography: can biopsy be deferred?](https://pubmed.ncbi.nlm.nih.gov/18855166/)
  4. [Palpable breast masses with probably benign morphology at sonography: can biopsy be deferred?](https://pubmed.ncbi.nlm.nih.gov/18855166/)
  5. [Shear-wave elastography and greyscale assessment of palpable probably benign masses: is biopsy always required?](https://pubmed.ncbi.nlm.nih.gov/27007593/)

### 3. Surgery (Acute Abdomen)

> I have this sharp pain in my lower right belly that gets worse when I move.

- Expected: Acute appendicitis (Right lower quadrant tenderness with rebound)
- Shown to the user: **insufficient evidence**
- Model's own answer: Insufficient evidence
- Evidence: not_supported; 1 attempt(s); 0.9 s
- Verdict: ⚪ declined (no diagnosis)

  - The provided passages do not discuss lower abdominal pain related to movement. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Facial pain radiating upwards: could the pain of epicrania fugax start in the lower face?](https://pubmed.ncbi.nlm.nih.gov/25532591/)
  2. [Does esophageal function vary at the striated and smooth muscle segments in functional chest pain?](https://pubmed.ncbi.nlm.nih.gov/12358233/)
  3. [Is there a relationship between preferred posture and positioning in early life and the direction of subsequent asymmetrical postural deformity in non ambulant people with cerebral palsy?](https://pubmed.ncbi.nlm.nih.gov/18796054/)
  4. [Pain, physical dependence and pseudoaddiction: redefining addiction for 'nice' people?](https://pubmed.ncbi.nlm.nih.gov/18768306/)
  5. [Is pelvic pain associated with defecatory symptoms in women with pelvic organ prolapse?](https://pubmed.ncbi.nlm.nih.gov/21394762/)

### 4. Diabetes / Endocrinology

> I’m always thirsty, peeing all the time, and losing weight without trying.

- Expected: Diabetes mellitus (likely type 1 if acute) (Polyuria, polydipsia, unexplained weight loss)
- Shown to the user: **insufficient evidence**
- Model's own answer: Your symptoms may suggest uncontrolled diabetes and you should seek medical evaluation promptly.
- Evidence: not_supported; 2 attempt(s); 2.0 s
- Verdict: ✅ match (uncontrolled diabetes)

  - Persistent thirst, polyuria, and unexplained weight loss are classic signs of diabetes mellitus (not found in the sources, support 0.00)
  - Early diagnosis and treatment are important to prevent complications (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Are weight-related attitudes and behaviours associated with the accuracy of BMI derived from self-reported weight and height among 13-year-olds?](https://pubmed.ncbi.nlm.nih.gov/25525037/)
  2. [Is alcohol drinking associated with renal impairment in the general population of South Korea?](https://pubmed.ncbi.nlm.nih.gov/24923262/)
  3. [Does a Vacation Break Impact the Outcomes of Required Clinical Clerkships?](https://pubmed.ncbi.nlm.nih.gov/27272424/)
  4. [Is alcohol drinking associated with renal impairment in the general population of South Korea?](https://pubmed.ncbi.nlm.nih.gov/24923262/)
  5. [Is perforation of the appendix a risk factor for tubal infertility and ectopic pregnancy?](https://pubmed.ncbi.nlm.nih.gov/10223070/)

### 5. Infection

> I’ve got a bad headache, stiff neck, and light really hurts my eyes.

- Expected: Meningitis (likely bacterial until proven otherwise) (Neck stiffness, photophobia, meningismus)
- Shown to the user: **insufficient evidence**
- Model's own answer: Your symptoms could be concerning for meningitis, which typically presents with headache, neck stiffness, and photophobia.
- Evidence: not_supported; 2 attempt(s); 1.9 s
- Verdict: ✅ match (meningitis)

  - Headache, neck stiffness, and photophobia are classic signs of meningitis (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Red fluorescence in reef fish: a novel signalling mechanism?](https://pubmed.ncbi.nlm.nih.gov/18796150/)
  2. [Facial pain radiating upwards: could the pain of epicrania fugax start in the lower face?](https://pubmed.ncbi.nlm.nih.gov/25532591/)
  3. [Does early migraine treatment shorten time to headache peak and reduce its severity?](https://pubmed.ncbi.nlm.nih.gov/18005142/)
  4. [Pseudomigraine with lymphocytic pleocytosis: a calcium channelopathy?](https://pubmed.ncbi.nlm.nih.gov/12940811/)
  5. [Does the treatment of amblyopia normalise subfoveal choroidal thickness in amblyopic children?](https://pubmed.ncbi.nlm.nih.gov/27757987/)

### 6. Ophthalmology

> Suddenly I can’t see out of one eye, like a curtain came down.

- Expected: Retinal detachment (Acute monocular vision loss (amaurosis/retinal detachment description))
- Shown to the user: **insufficient evidence**
- Model's own answer: Sudden vision loss in one eye described as a curtain coming down suggests retinal detachment and requires urgent ophthalmologic evaluation.
- Evidence: not_supported; 2 attempt(s); 2.0 s
- Verdict: ✅ match (retinal detachment)

  - Sudden unilateral vision loss described as a curtain is classic for retinal detachment (not found in the sources, support 0.00)
  - Retinal detachment is an ophthalmic emergency that needs prompt referral (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Fulfilling human resources development goal in West Africa: can the training of ophthalmologist diplomates be improved?](https://pubmed.ncbi.nlm.nih.gov/20306735/)
  2. [Are all aberrations equal?](https://pubmed.ncbi.nlm.nih.gov/12361157/)
  3. [Early postoperative outcomes following surgical repair of complete atrioventricular septal defects: is down syndrome a risk factor?](https://pubmed.ncbi.nlm.nih.gov/24201860/)
  4. [Convulsions and retinal haemorrhage: should we look further?](https://pubmed.ncbi.nlm.nih.gov/11970923/)
  5. [Spontaneous subarachnoid hemorrhage,: is emergency transcranial Doppler sonography useful?](https://pubmed.ncbi.nlm.nih.gov/14506552/)

### 7. Orthopaedics

> My hip hurts so much after I fell, and now I can’t stand on that leg.

- Expected: Femoral neck fracture (Inability to bear weight with suspected hip fracture)
- Shown to the user: **insufficient evidence**
- Model's own answer: You should have your hip evaluated promptly, including imaging (e.g., X‑ray) and orthopedic assessment, as severe pain and inability to bear weight after a fall may indicate a fracture or other serious injury.
- Evidence: not_supported; 2 attempt(s); 2.0 s
- Verdict: 🟡 partial (possible fracture, not specifically femoral neck)

  - Severe hip pain and inability to stand after a fall are red‑flag symptoms for possible fracture (not found in the sources, support 0.00)
  - Prompt imaging and orthopedic evaluation are standard care for suspected hip fractures (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Total hip arthroplasty performed in patients with residual poliomyelitis: does it work?](https://pubmed.ncbi.nlm.nih.gov/24203163/)
  2. [Total hip arthroplasty performed in patients with residual poliomyelitis: does it work?](https://pubmed.ncbi.nlm.nih.gov/24203163/)
  3. [Long-term results of reconstructive surgery in infantile cerebral palsy patients with high hip dislocation: is hip screening necessary?](https://pubmed.ncbi.nlm.nih.gov/24816981/)
  4. [Do support groups help people with chronic fatigue syndrome and fibromyalgia?](https://pubmed.ncbi.nlm.nih.gov/16331774/)
  5. [Does intact limb loading differ in servicemembers with traumatic lower limb loss?](https://pubmed.ncbi.nlm.nih.gov/24832826/)

### 8. Biochemistry / Metabolic

> I feel really weak, my heart feels weird, and my muscles aren’t working right.

- Expected: Hyperkalemia (Muscle weakness with cardiac arrhythmia symptoms)
- Shown to the user: **insufficient evidence**
- Model's own answer: Insufficient evidence
- Evidence: not_supported; 2 attempt(s); 1.7 s
- Verdict: ⚪ declined (no diagnosis)

  - The provided passages do not contain sufficient information to explain the combination of generalized weakness, cardiac symptoms, and muscle dysfunction described. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Is IVIg therapy warranted in progressive lower motor neuron syndromes without conduction block?](https://pubmed.ncbi.nlm.nih.gov/24212395/)
  2. [Do support groups help people with chronic fatigue syndrome and fibromyalgia?](https://pubmed.ncbi.nlm.nih.gov/16331774/)
  3. [Acute pure motor quadriplegia: is it dengue myositis?](https://pubmed.ncbi.nlm.nih.gov/16315973/)
  4. [Symptoms of hormonal therapy and social support: Is there a connection?](https://pubmed.ncbi.nlm.nih.gov/25529935/)
  5. [Does the habitual mastication side impact jaw muscle activity?](https://pubmed.ncbi.nlm.nih.gov/27019139/)

### 9. Clinical Outcomes / Critical Care

> I feel really sick, dizzy, and confused, and my family says I have a fever.

- Expected: Sepsis / septic shock (Altered mental status with fever and hypotension)
- Shown to the user: **insufficient evidence**
- Model's own answer: Fever is a defining feature of influenza‑like illness and systemic symptoms such as dizziness or confusion may occur, but clinical evaluation is required to determine the cause and appropriate treatment.
- Evidence: partially_supported; 3 attempt(s); 4.8 s
- Verdict: ❌ different (influenza-like illness, not sepsis)

  - Fever is a defining criterion of influenza‑like illness (ILI). (source [1], support 0.77)
    > "An office visit or hospitalization of a patient for ILI, defined as presence of fever (temperature>or = 37.8 degrees C) and cough, sore throat, or myalgia, along with the physician's clinical judgment of influenza."
  - Clinical assessment is needed to determine cause and appropriate treatment. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Do family physicians make good sentinels for influenza?](https://pubmed.ncbi.nlm.nih.gov/8111516/)
  2. [Routine chest radiography of children with cancer hospitalized for fever and neutropenia: is it really necessary?](https://pubmed.ncbi.nlm.nih.gov/9305718/)
  3. [Do support groups help people with chronic fatigue syndrome and fibromyalgia?](https://pubmed.ncbi.nlm.nih.gov/16331774/)
  4. [Do psychological factors predict occurrence of influenza-like symptoms in vaccinated elderly residents of a sheltered home?](https://pubmed.ncbi.nlm.nih.gov/16238856/)
  5. [Do family physicians make good sentinels for influenza?](https://pubmed.ncbi.nlm.nih.gov/8111516/)

### 10. Cardiovascular (Emergency)

> I suddenly felt a tearing pain in my chest that goes straight to my back.

- Expected: Aortic dissection (Tearing chest pain radiating to the back)
- Shown to the user: **insufficient evidence**
- Model's own answer: insufficient evidence
- Evidence: not_supported; 1 attempt(s); 1.0 s
- Verdict: ⚪ declined (no diagnosis)

  - The provided passages do not contain information about tearing chest pain radiating to the back. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Does high blood pressure reduce the risk of chronic low back pain?](https://pubmed.ncbi.nlm.nih.gov/24019262/)
  2. [Motor performance in chronic low back pain: is there an influence of pain-related cognitions?](https://pubmed.ncbi.nlm.nih.gov/21951591/)
  3. [Does esophageal function vary at the striated and smooth muscle segments in functional chest pain?](https://pubmed.ncbi.nlm.nih.gov/12358233/)
  4. [A prospective, multi-institutional comparative effectiveness study of lumbar spine surgery in morbidly obese patients: does minimally invasive transforaminal lumbar interbody fusion result in superior outcomes?](https://pubmed.ncbi.nlm.nih.gov/25535070/)
  5. [Is there an anatomic explanation for chest pain in patients with pulmonary sarcoidosis?](https://pubmed.ncbi.nlm.nih.gov/9305302/)

### 11. Cancer (Colorectal)

> I’ve been seeing blood in my poop and my pants feel looser lately.

- Expected: Colorectal cancer (Hematochezia with unintentional weight loss)
- Shown to the user: **Changes in bowel habits such as looser stools are recognized as symptoms that raise suspicion for colorectal cancer and other serious conditions.**
- Model's own answer: Changes in bowel habits such as looser stools are recognized as symptoms that raise suspicion for colorectal cancer and other serious conditions.
- Evidence: supported; 3 attempt(s); 4.5 s
- Verdict: 🟡 partial (names colorectal cancer, but reads 'pants feel looser' as looser stools and ignores the blood)

  - Diarrhoea is a factor that predicts recognition of cancer presentations. (source [3], support 0.98)
    > "Factors that predict recognition of cancer presentations include rectal bleeding, weight loss and diarrhoea."

  Retrieved sources:
  1. [Advice to consult a general medical practitioner in Western Australia: could it be cancer?](https://pubmed.ncbi.nlm.nih.gov/19281671/)
  2. [Do Effects on Bowel Patterns Imposed by Solifenacin Always Have Negative Impacts on Treating Patients with Overactive Bladder (OAB)?](https://pubmed.ncbi.nlm.nih.gov/27111622/)
  3. [Advice to consult a general medical practitioner in Western Australia: could it be cancer?](https://pubmed.ncbi.nlm.nih.gov/19281671/)
  4. [Is living in a rural area good for your respiratory health?](https://pubmed.ncbi.nlm.nih.gov/16236855/)
  5. [Are depressive symptoms associated with cancer screening and cancer stage at diagnosis among postmenopausal women?](https://pubmed.ncbi.nlm.nih.gov/18788983/)

### 12. Cancer (Lung)

> I keep coughing and sometimes there’s blood, and I’ve lost weight without trying.

- Expected: Lung cancer (Chronic cough with hemoptysis and cachexia)
- Shown to the user: **insufficient evidence**
- Model's own answer: Insufficient evidence
- Evidence: not_supported; 2 attempt(s); 2.3 s
- Verdict: ⚪ declined (no diagnosis)

  - The provided passages do not contain information linking coughing with blood and unexplained weight loss to a specific diagnosis. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Are weight-related attitudes and behaviours associated with the accuracy of BMI derived from self-reported weight and height among 13-year-olds?](https://pubmed.ncbi.nlm.nih.gov/25525037/)
  2. [Is living in a rural area good for your respiratory health?](https://pubmed.ncbi.nlm.nih.gov/16236855/)
  3. [Is endothelin-1 an aggravating factor in the development of acute pancreatitis?](https://pubmed.ncbi.nlm.nih.gov/11601252/)
  4. [Acute pure motor quadriplegia: is it dengue myositis?](https://pubmed.ncbi.nlm.nih.gov/16315973/)
  5. [Pseudomigraine with lymphocytic pleocytosis: a calcium channelopathy?](https://pubmed.ncbi.nlm.nih.gov/12940811/)

### 13. Surgery (Gallbladder)

> I get this bad pain on the right side under my ribs after eating greasy food.

- Expected: Cholelithiasis / cholecystitis (Right upper quadrant pain postprandial (biliary colic))
- Shown to the user: **insufficient evidence**
- Model's own answer: Your symptoms may suggest biliary colic or gallbladder disease, which can cause right-sided upper abdominal pain after fatty meals.
- Evidence: not_supported; 2 attempt(s); 2.3 s
- Verdict: ✅ match (biliary colic / gallbladder disease)

  - Right-sided upper abdominal pain after eating greasy food is characteristic of biliary colic (not found in the sources, support 0.00)
  - Gallbladder disease can present with pain under the ribs post‑prandially (not found in the sources, support 0.00)

  Retrieved sources:
  1. [School food policy at Dutch primary schools: room for improvement?](https://pubmed.ncbi.nlm.nih.gov/23587089/)
  2. [Is there an anatomic explanation for chest pain in patients with pulmonary sarcoidosis?](https://pubmed.ncbi.nlm.nih.gov/9305302/)
  3. [Is plate clearing a risk factor for obesity?](https://pubmed.ncbi.nlm.nih.gov/25521278/)
  4. [Dietary linoleic acid-induced alterations in pro- and anti-nociceptive lipid autacoids: Implications for idiopathic pain syndromes?](https://pubmed.ncbi.nlm.nih.gov/27030719/)
  5. [School food policy at Dutch primary schools: room for improvement?](https://pubmed.ncbi.nlm.nih.gov/23587089/)

### 14. Surgery (Hernia)

> There’s a bulge in my groin that pops out when I cough or lift things.

- Expected: Inguinal hernia (Reducible inguinal mass with Valsalva)
- Shown to the user: **insufficient evidence**
- Model's own answer: insufficient evidence
- Evidence: not_supported; 2 attempt(s); 2.2 s
- Verdict: ⚪ declined (no diagnosis)

  - The provided passages do not describe the clinical presentation of groin hernia as a bulge that appears with coughing or lifting. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Is endoscopic totally extraperitoneal hernioplasty justified for the repair of groin hernia in female patients?](https://pubmed.ncbi.nlm.nih.gov/16308798/)
  2. [Is endoscopic totally extraperitoneal hernioplasty justified for the repair of groin hernia in female patients?](https://pubmed.ncbi.nlm.nih.gov/16308798/)
  3. [Are voiding symptoms really associated with abnormal urodynamic voiding parameters in women?](https://pubmed.ncbi.nlm.nih.gov/19054175/)
  4. [Is minilaparoscopic inguinal hernia repair feasible?](https://pubmed.ncbi.nlm.nih.gov/15954832/)
  5. [Is polypropylene mesh safe and effective for repairing infected incisional hernia in renal transplant recipients?](https://pubmed.ncbi.nlm.nih.gov/16230159/)

### 15. Diabetes / Endocrinology

> My vision gets blurry sometimes and I feel tired all the time.

- Expected: Type 2 diabetes mellitus (Intermittent blurred vision with fatigue (hyperglycemia-related))
- Shown to the user: **insufficient evidence**
- Model's own answer: Insufficient evidence to determine the cause of blurry vision and fatigue.
- Evidence: not_supported; 2 attempt(s); 1.7 s
- Verdict: ⚪ declined (no diagnosis)

  - The provided passages do not link blurry vision or chronic fatigue to a specific condition. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Do older adults with cancer fall more often?](https://pubmed.ncbi.nlm.nih.gov/23448747/)
  2. [Does visual impairment lead to additional disability in adults with intellectual disabilities?](https://pubmed.ncbi.nlm.nih.gov/18771511/)
  3. [Affect-regulated exercise intensity: does training at an intensity that feels 'good' improve physical health?](https://pubmed.ncbi.nlm.nih.gov/22658587/)
  4. [Night-time frequency, sleep disturbance and general health-related quality of life: is there a relation?](https://pubmed.ncbi.nlm.nih.gov/19054172/)
  5. [Malpractice in invasive cardiology: is angiography of abdominal aorta or subclavian artery appropriate in patients undergoing coronary angiography?](https://pubmed.ncbi.nlm.nih.gov/16322917/)

### 16. Cardiovascular (Heart Failure)

> I can’t lie flat at night because I feel like I’m drowning.

- Expected: Congestive heart failure (Orthopnea)
- Shown to the user: **insufficient evidence**
- Model's own answer: insufficient evidence
- Evidence: not_supported; 2 attempt(s); 1.7 s
- Verdict: ⚪ declined (no diagnosis)

  - There is no provided passage that addresses symptoms of feeling like drowning when lying flat at night. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Affect-regulated exercise intensity: does training at an intensity that feels 'good' improve physical health?](https://pubmed.ncbi.nlm.nih.gov/22658587/)
  2. [Facial pain radiating upwards: could the pain of epicrania fugax start in the lower face?](https://pubmed.ncbi.nlm.nih.gov/25532591/)
  3. [Does magnifying narrow-band imaging or magnifying chromoendoscopy help experienced endoscopists assess invasion depth of large sessile and flat polyps?](https://pubmed.ncbi.nlm.nih.gov/24839918/)
  4. [Elderly suicide and weather conditions: is there a link?](https://pubmed.ncbi.nlm.nih.gov/9309473/)
  5. [Does pediatric housestaff experience influence tests ordered for infants in the neonatal intensive care unit?](https://pubmed.ncbi.nlm.nih.gov/9142039/)

### 17. Cardiovascular (Arrhythmia)

> My heart suddenly starts racing and feels like it’s skipping beats.

- Expected: Atrial fibrillation (Palpitations with irregular rhythm)
- Shown to the user: **insufficient evidence**
- Model's own answer: Ventricular premature beats and other ventricular arrhythmias can cause a rapid, irregular heart sensation.
- Evidence: partially_supported; 3 attempt(s); 4.3 s
- Verdict: ❌ different (ventricular arrhythmias, not atrial fibrillation)

  - Ventricular premature beats are a type of ventricular arrhythmia observed during exercise testing (not found in the sources, support 0.00)
  - Ventricular couplets and non‑sustained ventricular tachycardia are other ventricular arrhythmias detected in athletes (source [1], support 0.97)
    > "102 athletes had ventricular couplets or non-sustained ventricular tachycardia during an exercise test (group B)."

  Retrieved sources:
  1. [Do ventricular arrhythmias in athletes subside over time?](https://pubmed.ncbi.nlm.nih.gov/24340838/)
  2. [Are ATP sensitive potassium channels involved in the pronounced antiarrhythmic effects of preconditioning?](https://pubmed.ncbi.nlm.nih.gov/8324798/)
  3. [Effects of hypertrophy on regional action potential characteristics in the rat left ventricle: a cellular basis for T-wave inversion?](https://pubmed.ncbi.nlm.nih.gov/9323099/)
  4. [Can we use the 6-minute step test instead of the 6-minute walking test?](https://pubmed.ncbi.nlm.nih.gov/27012823/)
  5. [Does atrial fibrillation beget ventricular fibrillation in patients with acute myocardial infarction?](https://pubmed.ncbi.nlm.nih.gov/19067815/)

### 18. Infection (Urinary)

> It burns when I pee and I feel like I have to go all the time.

- Expected: Urinary tract infection (Dysuria with urinary frequency)
- Shown to the user: **insufficient evidence**
- Model's own answer: The provided passages do not contain information about burning during urination or a frequent urge to urinate.
- Evidence: not_supported; 2 attempt(s); 1.7 s
- Verdict: ⚪ declined (no diagnosis)

  - None of the passages mention dysuria or urinary frequency. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Community-based stroke information for clients with stroke and their carers: is there congruency between actual and recommended practice?](https://pubmed.ncbi.nlm.nih.gov/18782734/)
  2. [Do proxies reflect patients' health concerns about urinary incontinence and gait problems?](https://pubmed.ncbi.nlm.nih.gov/16305748/)
  3. [Body mass index and acute coronary syndromes: paradox or confusion?](https://pubmed.ncbi.nlm.nih.gov/24923461/)
  4. [Can bladder outflow obstruction be diagnosed from pressure flow analysis of voiding initiated by involuntary detrusor overactivity?](https://pubmed.ncbi.nlm.nih.gov/14501732/)
  5. [Is family history of depression a risk factor for poststroke depression?](https://pubmed.ncbi.nlm.nih.gov/19307856/)

### 19. Infection (Respiratory)

> I’ve had a fever and chills, and my chest hurts when I breathe in.

- Expected: Pneumonia (Fever with pleuritic chest pain)
- Shown to the user: **insufficient evidence**
- Model's own answer: Your symptoms of fever, chills, and chest pain on breathing may indicate a respiratory infection such as pneumonia and should be evaluated promptly, especially if you have neutropenia.
- Evidence: partially_supported; 2 attempt(s); 2.7 s
- Verdict: ✅ match (pneumonia)

  - Fever and neutropenia episodes can be associated with pneumonia documented by chest radiograph (source [3], support 0.91)
    > "Pneumonia was documented by chest radiograph in 4 of the 108 episodes (3.7%) of fever and neutropenia."
  - Chest pain on inspiration can be a sign of pleuritic involvement in respiratory infection (not found in the sources, support 0.10)

  Retrieved sources:
  1. [Routine chest radiography of children with cancer hospitalized for fever and neutropenia: is it really necessary?](https://pubmed.ncbi.nlm.nih.gov/9305718/)
  2. [Do family physicians make good sentinels for influenza?](https://pubmed.ncbi.nlm.nih.gov/8111516/)
  3. [Routine chest radiography of children with cancer hospitalized for fever and neutropenia: is it really necessary?](https://pubmed.ncbi.nlm.nih.gov/9305718/)
  4. [Is there an anatomic explanation for chest pain in patients with pulmonary sarcoidosis?](https://pubmed.ncbi.nlm.nih.gov/9305302/)
  5. [Routine chest radiography of children with cancer hospitalized for fever and neutropenia: is it really necessary?](https://pubmed.ncbi.nlm.nih.gov/9305718/)

### 20. Ophthalmology

> I see halos around lights and my eye hurts really badly.

- Expected: Acute angle-closure glaucoma (Halos with ocular pain and increased intraocular pressure)
- Shown to the user: **insufficient evidence**
- Model's own answer: Insufficient evidence
- Evidence: not_supported; 2 attempt(s); 3.7 s
- Verdict: ⚪ declined (no diagnosis)

  - The provided passages do not contain information about halos around lights or severe eye pain. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [Refractive errors in 3-6 year-old Chinese children: a very low prevalence of myopia?](https://pubmed.ncbi.nlm.nih.gov/24205064/)
  2. [Are higher blood mercury levels associated with dry eye symptoms in adult Koreans?](https://pubmed.ncbi.nlm.nih.gov/27121705/)
  3. [Is there a uniform basal endometrial gene expression profile during the implantation window in women who became pregnant in a subsequent ICSI cycle?](https://pubmed.ncbi.nlm.nih.gov/19542542/)
  4. [Facial pain radiating upwards: could the pain of epicrania fugax start in the lower face?](https://pubmed.ncbi.nlm.nih.gov/25532591/)
  5. [Courses of substance use and schizophrenia in the dual-diagnosis patients: is there a relationship?](https://pubmed.ncbi.nlm.nih.gov/12944690/)

### 21. Ophthalmology

> Everything looks blurry in the center, but I can still see around it.

- Expected: Age-related macular degeneration (Central vision loss (macular involvement))
- Shown to the user: **insufficient evidence**
- Model's own answer: insufficient evidence
- Evidence: no_evidence; 2 attempt(s); 3.6 s
- Verdict: ⚪ declined (no diagnosis)


  Retrieved sources:
  1. [Does escalation of the apical dose change treatment outcome in beta-radiation of posterior choroidal melanomas with 106Ru plaques?](https://pubmed.ncbi.nlm.nih.gov/11955750/)
  2. [Amblyopia: is visual loss permanent?](https://pubmed.ncbi.nlm.nih.gov/10966943/)
  3. [Can clinicians use the PHQ-9 to assess depression in people with vision loss?](https://pubmed.ncbi.nlm.nih.gov/19156007/)
  4. [Are antioxidants or supplements protective for age-related macular degeneration?](https://pubmed.ncbi.nlm.nih.gov/8311777/)
  5. [Are elevated homocysteine plasma levels related to peripheral arterial disease?](https://pubmed.ncbi.nlm.nih.gov/12925033/)

### 22. Orthopaedics

> My knee suddenly got swollen and painful after I twisted it.

- Expected: Anterior cruciate ligament (ACL) tear (Acute knee effusion after rotational injury)
- Shown to the user: **insufficient evidence**
- Model's own answer: You may have an acute ligament injury such as an ACL tear, which often causes sudden swelling and pain after a twisting motion.
- Evidence: partially_supported; 2 attempt(s); 3.3 s
- Verdict: ✅ match (ACL tear)

  - A twisting injury to the knee can cause ligament damage leading to swelling and pain (not found in the sources, support 0.00)
  - ACL injuries are commonly evaluated with laxity measurements and functional scores (source [2], support 0.85)
    > "Patients were divided into three groups according to the anterior laxity of the contralateral, normal knee ... Anterior laxity of the reconstructed knee was assessed at twenty-four months postoperatively. Anterior stability of the knee was also assessed with use of the Lachman and pivot-shift tests. Functional outcomes were assessed with the Lysholm score and the International Knee Documentation Committee (IKDC) score."

  Retrieved sources:
  1. [Can normal knee kinematics be restored with unicompartmental knee replacement?](https://pubmed.ncbi.nlm.nih.gov/15687156/)
  2. [Does anterior laxity of the uninjured knee influence clinical outcomes of ACL reconstruction?](https://pubmed.ncbi.nlm.nih.gov/24695920/)
  3. [Does elective re-siting of intravenous cannulae decrease peripheral thrombophlebitis?](https://pubmed.ncbi.nlm.nih.gov/19852337/)
  4. [Can Flexible Instruments Create Adequate Femoral Tunnel Lengths at 90° of Knee Flexion in Anterior Cruciate Ligament Reconstruction?](https://pubmed.ncbi.nlm.nih.gov/27281318/)
  5. [Is DAS28 an appropriate tool to assess remission in rheumatoid arthritis?](https://pubmed.ncbi.nlm.nih.gov/15941836/)

### 23. Orthopaedics

> My back pain shoots down my leg like an electric shock.

- Expected: Lumbar disc herniation (Radicular pain (sciatica))
- Shown to the user: **insufficient evidence**
- Model's own answer: Your description is characteristic of sciatica, which is often caused by lumbar nerve root irritation.
- Evidence: not_supported; 2 attempt(s); 8.5 s
- Verdict: 🟡 partial (sciatica from nerve root irritation; does not name disc herniation)

  - Electric shock-like leg pain is typical of nerve root involvement such as sciatica. (not found in the sources, support 0.00)

  Retrieved sources:
  1. [A prospective, multi-institutional comparative effectiveness study of lumbar spine surgery in morbidly obese patients: does minimally invasive transforaminal lumbar interbody fusion result in superior outcomes?](https://pubmed.ncbi.nlm.nih.gov/25535070/)
  2. [Does high blood pressure reduce the risk of chronic low back pain?](https://pubmed.ncbi.nlm.nih.gov/24019262/)
  3. [Assessing Patient Reported Outcomes Measures via Phone Interviews Versus Patient Self-Survey in the Clinic: Are We Measuring the Same Thing?](https://pubmed.ncbi.nlm.nih.gov/26548832/)
  4. [Do age and comorbidity affect quality of life or PTA-induced quality-of-life improvements in patients with symptomatic pad?](https://pubmed.ncbi.nlm.nih.gov/15943516/)
  5. [Psychosocial factors and surgical outcomes: are elderly depressed patients less satisfied with surgery?](https://pubmed.ncbi.nlm.nih.gov/24921847/)

### 24. Biochemistry / Metabolic

> My hands cramp up and my face feels twitchy.

- Expected: Hypocalcemia (Tetany (neuromuscular irritability))
- Shown to the user: **insufficient evidence**
- Model's own answer: Insufficient evidence
- Evidence: no_evidence; 2 attempt(s); 3.2 s
- Verdict: ⚪ declined (no diagnosis)


  Retrieved sources:
  1. [Does oral magnesium substitution relieve pregnancy-induced leg cramps?](https://pubmed.ncbi.nlm.nih.gov/18768245/)
  2. [Are reports of mechanical dysfunction in chronic oro-facial pain related to somatisation?](https://pubmed.ncbi.nlm.nih.gov/17919952/)
  3. [Is it cost-effective to require recreational ice hockey players to wear face protection?](https://pubmed.ncbi.nlm.nih.gov/18791529/)
  4. [Does early migraine treatment shorten time to headache peak and reduce its severity?](https://pubmed.ncbi.nlm.nih.gov/18005142/)
  5. [Do support groups help people with chronic fatigue syndrome and fibromyalgia?](https://pubmed.ncbi.nlm.nih.gov/16331774/)

### 25. Clinical Outcomes / Shock

> I feel faint, sweaty, and like I might pass out after losing a lot of blood.

- Expected: Hypovolemic shock (Hypotension with signs of hypoperfusion)
- Shown to the user: **insufficient evidence**
- Model's own answer: Your symptoms are likely due to low blood pressure from significant blood loss (hypovolemia).
- Evidence: not_supported; 2 attempt(s); 3.0 s
- Verdict: ✅ match (hypovolemia from blood loss)

  - Significant blood loss can cause hypovolemia leading to hypotension (not found in the sources, support 0.02)
  - Hypotension can produce faintness, sweating, and a feeling of impending syncope (not found in the sources, support 0.01)

  Retrieved sources:
  1. [Does an interdisciplinary network improve dementia care?](https://pubmed.ncbi.nlm.nih.gov/24938504/)
  2. [Acute pure motor quadriplegia: is it dengue myositis?](https://pubmed.ncbi.nlm.nih.gov/16315973/)
  3. [Affect-regulated exercise intensity: does training at an intensity that feels 'good' improve physical health?](https://pubmed.ncbi.nlm.nih.gov/22658587/)
  4. [Continuous spinal anesthesia: does low-dose plain or hyperbaric bupivacaine allow the performance of hip surgery in the elderly?](https://pubmed.ncbi.nlm.nih.gov/8323890/)
  5. [Are specific residency program characteristics associated with the pass rate of graduates on the ABFM certification examination?](https://pubmed.ncbi.nlm.nih.gov/24915479/)
