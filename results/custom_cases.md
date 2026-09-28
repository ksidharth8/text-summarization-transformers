# Custom test cases — bart-base fine-tuned on CNN/DailyMail

### 01_news_electric_buses
*242 words / 285 tokens — single pass, adaptive length — 0.8s — novel bigrams 2.63%*

**Abstractive (bart-base fine-tuned):** The municipal corporation of a mid-sized hill city has approved a plan to replace its entire diesel bus fleet with 120 electric buses by the end of 2028. The first batch of 30 buses is expected to arrive in March and will run on the four busiest routes. The project will cost an estimated 210 crore rupees, with roughly half the money coming from a national clean-transport grant and the rest from a low-interest loan.

**TextRank:** The municipal corporation of a mid-sized hill city has approved a plan to replace its entire diesel bus fleet with 120 electric buses by the end of 2028, officials said on Monday. Transport officials said the switch would cut the corporation's fuel bill by nearly 40 per cent and remove around 9,000 tonnes of carbon dioxide emissions every year. The corporation said a trial with two leased buses had already covered 18,000 kilometres without a single breakdown, and that the tender requires manufacturers to guarantee a range of at least 220 kilometres on a single charge under full load.

**LEAD-3:** The municipal corporation of a mid-sized hill city has approved a plan to replace its entire diesel bus fleet with 120 electric buses by the end of 2028, officials said on Monday. The first batch of 30 buses is expected to arrive in March and will run on the four busiest routes, which together carry about 65,000 passengers a day. The project will cost an estimated 210 crore rupees, with roughly half the money coming from a national clean-transport grant and the rest from a low-interest loan.

### 02_news_flood_warning_system
*252 words / 299 tokens — single pass, adaptive length — 0.7s — novel bigrams 3.51%*

**Abstractive (bart-base fine-tuned):** The system combines rainfall readings from 140 automatic weather stations with water-level sensors installed on 26 bridges and embankments. A hydrological model updates forecasts every hour and sends alerts in Hindi, English and two tribal languages to registered mobile numbers in the affected blocks. Officials said the pilot ran in three districts during last year's monsoon.

**TextRank:** Villages along a flood-prone river basin in eastern India will receive flood warnings by SMS and voice call up to 36 hours in advance under a new early-warning system launched by the state disaster management authority on Thursday. Officials said the pilot, which ran in three districts during last year's monsoon, sent 41 alerts and correctly predicted 37 of the 39 floods that occurred, with an average lead time of 22 hours. A hydrologist who reviewed the pilot said the authority should publish its accuracy figures every season and explain every false alarm publicly, so that villagers do not start ignoring the messages.

**LEAD-3:** Villages along a flood-prone river basin in eastern India will receive flood warnings by SMS and voice call up to 36 hours in advance under a new early-warning system launched by the state disaster management authority on Thursday. The system combines rainfall readings from 140 automatic weather stations with water-level sensors installed on 26 bridges and embankments. A hydrological model updates forecasts every hour and sends alerts in Hindi, English and two tribal languages to registered mobile numbers in the affected blocks.

### 03_research_abstract
*184 words / 233 tokens — single pass, adaptive length — 0.7s — novel bigrams 1.72%*

**Abstractive (bart-base fine-tuned):** Large language models are increasingly deployed to summarize scientific articles, yet their summaries frequently contain statements that are not supported by the source document. We study how the length of the input context affects the factual consistency of abstractive summaries of biomedical papers. We construct a benchmark of 2,400 open-access articles paired with expert-written lay summaries.

**TextRank:** We construct a benchmark of 2,400 open-access articles paired with expert-written lay summaries and annotate 9,600 model-generated sentences for unsupported claims. We compare three strategies for handling documents that exceed the model's context window: truncation to the first 1,024 tokens, retrieval of the most relevant sections, and hierarchical summarization in which each section is summarized independently before a final summary is produced. Hierarchical summarization improves ROUGE-L by 2.1 points over truncation and reduces the rate of unsupported sentences from 18.4 per cent to 11.7 per cent, whereas retrieval gives smaller gains on both measures.

**LEAD-3:** Large language models are increasingly deployed to summarize scientific articles, yet their summaries frequently contain statements that are not supported by the source document. In this work we study how the length of the input context affects the factual consistency of abstractive summaries of biomedical papers. We construct a benchmark of 2,400 open-access articles paired with expert-written lay summaries and annotate 9,600 model-generated sentences for unsupported claims.

### 04_long_paper_token_pruning
*838 words / 1037 tokens — long-document map-reduce: 3 chunks, 1 round(s) — 1.9s — novel bigrams 5.26%*

**Abstractive (bart-base fine-tuned):** Adaptive Token Pruning for Efficient Transformer Inference on Edge Devices. Transformer models achieve strong accuracy on language and vision tasks, but their inference cost grows quadratically with the number of input tokens. Adaptive token pruning offers a simple way to make transformer inference cheaper on edge devices. Future work will extend the approach to encoder-decoder models and combine it with weight quantization.

**TextRank:** We propose an adaptive token pruning method that removes uninformative tokens layer by layer, using a lightweight scoring head trained jointly with the model. On the text benchmarks, adaptive pruning reduces average latency by 38 per cent on the smartphone and 44 per cent on the single-board computer, with an average accuracy drop of 0.6 percentage points. On the image benchmark, adaptive pruning removes on average 61 per cent of the patches by the final layer and reduces latency by 52 per cent with a top-1 accuracy drop of 0.9 points.

**LEAD-3:** Adaptive Token Pruning for Efficient Transformer Inference on Edge Devices Transformer models achieve strong accuracy on language and vision tasks, but their inference cost grows quadratically with the number of input tokens, which makes deployment on phones and embedded boards difficult. We propose an adaptive token pruning method that removes uninformative tokens layer by layer, using a lightweight scoring head trained jointly with the model.

<details><summary>partial (chunk) summaries</summary>

1. Adaptive Token Pruning for Efficient Transformer Inference on Edge Devices. Transformer models achieve strong accuracy on language and vision tasks, but their inference cost grows quadratically with the number of input tokens. We propose an adaptive token pruning method that removes uninformative tokens layer by layer, using a lightweight scoring head trained jointly with the model. On three text classification benchmarks and one image classification benchmark, the method reduces inference latency by 38 to 52 per cent
2. We evaluate the method on three text benchmarks covering sentiment analysis, topic classification and natural language inference. For text we fine-tune a twelve-layer encoder with 110 million parameters. For images we use a vision transformer of similar size that splits each image into 196 patches. Latency is measured on a mid-range smartphone processor and on a single-board computer.
3. Adaptive token pruning offers a simple way to make transformer inference cheaper on edge devices. Future work will extend the approach to encoder-decoder models and combine it with weight quantization.
</details>
