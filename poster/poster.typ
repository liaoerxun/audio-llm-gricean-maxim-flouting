#import "@preview/peace-of-posters:0.6.0": *

#set page("a0", margin: 1.5cm)
#set-poster-layout(layout-a0)
#set text(font: "EB Garamond 12", size: 1.04 * layout-a0.at("body-size"))



#let leiden-blue = rgb("#06123B")
#let leiden-light-blue = rgb("#DFE3EE")

#let leiden-theme = (
  "body-box-args": (
    inset: 0.6em,
    width: 100%,
    stroke: leiden-blue,
  ),
  "body-text-args": (
    fill: rgb("#111827"),
  ),
  "heading-box-args": (
    inset: 0.6em,
    width: 100%,
    fill: leiden-blue,
    stroke: leiden-blue,
  ),
  "heading-text-args": (
    fill: white,
    weight: "medium",
  ),
  "title-box-args": (
    inset: 0.8em,
    width: 100%,
    fill: leiden-blue,
    stroke: leiden-blue,
  ),
  "title-text-args": (
    fill: white,
  ),
)

#set-theme(leiden-theme)



#update-poster-layout(column-spacing: 1.5cm)

#title-box(
  [Decoding Pragmatic Cues:],
  subtitle: [Evaluating Audio-Enabled Large Language Models in Detecting Gricean Maxim Flouting],
  authors: [Alexis Liao],
  institutes: [
    Leiden University · 
    #link("mailto:e.liao.2@umail.leidenuniv.nl")[
      e.liao.2\@umail.leidenuniv.nl
    ]
  ],
  keywords: [Audio LLMs · Pragmatics · Gricean maxims · Paralinguistic cues],
)

#v(0.4cm)

#columns(3, [

  // ── Column 1 ──────────────────────────────────────────────────────────────

  #column-box(heading: [Background])[
    Paralinguistic features — pitch, stress, timing, filler words, hedges, and voice quality — can alter how an utterance is pragmatically perceived.

    These cues may help listeners detect whether a speaker is deliberately *flouting* a conversational maxim and thereby conveying a conversational implicature. In accordance with Grice, such cues serve as supplementary evidence for pragmatic inference.

    Audio-enabled language models can process speech directly, bypassing transcription. However, it remains unclear whether their pragmatic judgments are genuinely sensitive to paralinguistic contrasts or merely reflect model-specific response preferences.
  ]


  #column-box(heading: [Research Questions])[
    #set text(size: 1.08em)
  
    *RQ1.* Can audio-enabled language models use paralinguistic contrasts to distinguish literal readings from conversational implicatures?
  
    #v(0.5em)
  
    *RQ2.* Can paired multimodal demonstrations and explicit prosodic guidance improve condition-sensitive, maxim-specific pragmatic reasoning?
  
    #set text(size: 1em)
    #v(0.5em)

    We examine Quality, Quantity, and Manner. Relation was excluded because its interpretation depends primarily on discourse context rather than the manipulated paralinguistic contrast.
  ]


  #column-box(heading: [Methods])[
    #set text(size: 0.88em)

    === Stimuli
    - 45 target items: 15 each for Quality, Quantity, and Manner
    - Each item included context, a preceding utterance, and matched recordings:
      - *Neutral* — literal reading expected
      - *Marked* — implicature and maxim flouting expected
    - A transcript-only condition provided a lexical baseline
    - Stimuli underwent blind auditing, revision, novelty checking, and maxim relabelling

    === Models
    `Qwen2-Audio-7B-Instruct` · `Kimi-Audio-7B-Instruct` · `Step-Audio-2-mini`

    === Prompting Regimes
    1. *Zero-shot*
    2. *Paired 6-shot*: neutral and marked recordings from one development item per maxim
    3. *Prosody-informed 6-shot*: the same demonstrations plus explicit, non-deterministic cue–pragmatics guidance

    Follow-up demonstrations used three held-out pilot items; all 45 target items remained unseen.

    === Evaluation
    Reading accuracy · Maxim accuracy · Joint accuracy · Correct neutral→marked item pairs · Exact paired tests
  ]

  #colbreak()

  // ── Column 2 ──────────────────────────────────────────────────────────────

  #column-box(heading: [Zero-shot Results])[
    Single-condition accuracy was strongly shaped by model-specific response preferences.

    - *Kimi-Audio* predicted a literal reading for 93.3 % of marked trials and selected `none` as the maxim for all 135 trials.
    - *Qwen2-Audio* predicted an implicature for every trial and selected `quality` for almost all audio trials.
    - *Step-Audio 2* predicted an implicature for every marked trial, but frequently paired `implicature` with `maxim = none`.

    Marked-condition reading accuracy was therefore *misleadingly high* for Qwen2-Audio and Step-Audio 2, since both models also over-predicted implicatures in the neutral condition.

    === Condition-Sensitive Performance

    #set text(size: 0.88em)
    #table(
      columns: (auto, 1fr, 1fr, 1fr),
      inset: 6pt,
      align: (left, center, center, center),
      stroke: 0.5pt,
      table.header(
        [*Model*], [*Neutral acc.*], [*Marked acc.*], [*Correct pairs*],
      ),
      [Kimi-Audio],    [97.8 %], [6.7 %],  [2/45 (4.4 %)],
      [Qwen2-Audio],   [0.0 %],  [100 %],  [0/45 (0.0 %)],
      [Step-Audio 2],  [11.1 %], [100 %],  [5/45 (11.1 %)],
    )
    #set text(size: layout-a0.at("body-size"))

    #v(0.5em)
    No model showed significant zero-shot paired sensitivity:

    #set text(size: 0.9em)
    Kimi: 2/45 correct reading pairs, _p_ = .50 ·
    Qwen: 0/45, _p_ = 1.00 ·
    Step-Audio: 5/45, _p_ = .0625
  ]

  #column-box(heading: [Implicature Rate by Condition])[
    #image("assets/implicature_rate_by_condition.png", width: 100%)
  ]

  #column-box(heading: [Zero-shot Failure Profiles])[
    #set text(size: 0.92em)

    *Kimi — `literal + none`*\
    Selected `none` for every trial and rarely changed its reading.

    *Qwen — `implicature + quality`*\
    Applied nearly identical labels across neutral and marked audio.

    *Step-Audio — `implicature + none`*\
    Often predicted implicature without identifying a flouted maxim.

    These profiles suggest that *implicature detection and maxim identification are separable capabilities*.
  ]

  #colbreak()

  // ── Column 3 ──────────────────────────────────────────────────────────────

  #column-box(heading: [Prompting Follow-up])[
    #set text(size: 0.92em)

    We tested whether matched demonstrations and explicit prosodic knowledge could overcome zero-shot response biases.

    #image(
      "assets/prompting_regime_comparison.svg",
      width: 100%,
    )

    #v(0.4em)

    *Kimi-Audio was the only model to benefit from paired demonstrations.*

    - Correct reading pairs increased from *2/45 to 29/45* (_p_ < .001).
    - Correct reading + maxim pairs increased from *0/45 to 11/45* (_p_ < .001).
    - Explicit prosodic guidance reduced reading pairs to *7/45* (_p_ < .001 vs. paired 6-shot).
  ]

  #column-box(heading: [Model-specific Prompt Effects])[
    #set text(size: 0.92em)

    *Qwen* remained almost categorically biased toward `implicature + quality` under all three regimes.

    *Kimi* learned the neutral–marked reading contrast from paired demonstrations, but continued to over-select Quality. Explicit guidance shifted it back toward `literal + none`.

    *Step-Audio* copied label alternatives in the paired 6-shot condition, then collapsed to `literal + none` on all 90 prosody-informed trials.

    Prompting changed *response biases* more readily than it improved maxim-specific pragmatic reasoning.
  ]
  #column-box(heading: [Conclusion])[
    #set text(size: 0.84em)

    *Zero-shot performance did not demonstrate reliable, maxim-specific sensitivity to the neutral–marked contrast.* Single-condition accuracy largely reflected model-specific response biases rather than pragmatic competence.

    Paired multimodal demonstrations substantially improved Kimi-Audio, increasing correct reading pairs from *2/45 to 29/45*. However, maxim identification remained strongly biased toward Quality.

    Explicit prosodic guidance provided no additional benefit: Qwen remained unchanged, while Kimi and Step-Audio shifted toward conservative `literal + none` responses.

    These findings show that audio access, examples, and explicit cue descriptions do not guarantee robust integration of paralinguistic evidence with pragmatic reasoning.

    #v(0.4em)
    #set text(style: "italic", size: 1.06em)
    More explicit prompting changed model biases more readily than it improved pragmatic reasoning.

    #v(0.6em)
    #set text(style: "normal", size: 0.82em)
    *Limitations:* one speaker, no independent human validation, three open-weight models, three maxim categories, and one fixed demonstration set. Future work should add multiple speakers, human judgments, acoustic measurements, and controlled cue manipulations. True fine-tuning is also expected.
  ]

])
