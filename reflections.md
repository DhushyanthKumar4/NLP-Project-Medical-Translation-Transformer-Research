Reflection — Domain-Specific Neural Machine Translation



🔹 Where the model fails (critical insight)

The model exhibits severe failure modes, despite seemingly good training loss:

❌ 1. Catastrophic semantic failure
Outputs unrelated languages (German, French)
Produces nonsensical tokens:
"141."
"1501 1 100"

Indicates the model is not learning stable language mapping.

❌ 2. Hallucination & corruption

From your metrics:

Numerical mismatch rate: 43.4%
Terminology preservation: 0.0

This is extremely critical in domains like medicine.

Example:

Input: structured technical sentence
Output: random symbols or unrelated language
❌ 3. BLEU collapse
Baseline BLEU: 7.72
Final BLEU: ~0.0

Performance degradation after “improvements” signals training instability or misconfiguration

❌ 4. Degenerate generation
Very short outputs
Repeated tokens
Empty outputs




🔹 Surprising domain behaviors
⚠️ 1. Cross-lingual drift

Model outputs:

German
French

instead of Korean

Suggests pretrained multilingual interference without proper fine-tuning constraints

⚠️ 2. Domain sensitivity

Better performance in:

외과학 (Surgery)
반도체 (Semiconductors)

Worse in:

병리학 (Pathology)

Indicates uneven domain representation in training data

⚠️ 3. Scaling instability

From scaling curve:

Data %	BLEU
25%	4.20
100%	1.91

More data reduced performance, which is counterintuitive

Likely causes:

Noisy data
Domain mixing
Overfitting or catastrophic forgetting




🔹 Data scaling limits

This project clearly shows:

More data ≠ better performance

Problems observed:

Data heterogeneity
Label noise
Domain mismatch




🔹 Real-world deployment concerns

This model is NOT safe for deployment, especially in medical contexts.

🚨 Critical risks:
Incorrect medical instructions
Numerical errors (dosage, measurements)
Language switching mid-sentence

This could directly lead to harmful outcomes




🔹 Ethical implications (very important)

In medical translation:

⚠️ 1. Patient safety risk
Incorrect translation → wrong treatment
⚠️ 2. Trust erosion
Users assume correctness from AI
⚠️ 3. Bias amplification
Domain imbalance → unequal quality




🔹 What went wrong (deep insight)

Your logs strongly suggest:

Root causes:
Inadequate fine-tuning strategy
Tokenization mismatch
Domain shift
Lack of constraint during generation
Possibly wrong training objective alignment




🔹 What I would fix (research-level improvements)
🧠 1. Use domain-adapted models
Fine-tune specifically for medical Korean
📚 2. Clean dataset aggressively
Remove noisy samples
Filter mixed-language data
🔧 3. Constrain decoding
Force Korean vocabulary
Penalize unknown tokens
📊 4. Better evaluation
Use:
COMET
BLEURT
instead of only BLEU
🤖 5. Try stronger architectures
T5-large / mT5
Instruction-tuned models
🔒 6. Add safety layer
Confidence scoring
Reject low-quality outputs




🔹 Final Takeaway

This project highlights that training neural machine translation systems in specialized domains is highly sensitive to data quality, domain alignment, and decoding constraints, and that naive scaling or tuning can degrade performance rather than improve it.
