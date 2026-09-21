# Glossary of Terms and Concepts in the Paper

> Companion to the paper *Organic Operations on Neural Latent Spaces: Interaction Paradigms and Aesthetic Boundaries of AI-Assisted Botanica/IDM Timbre Design* (v6 and later; section numbers §n follow the paper).
> 中文版：[glossary.zh.md](glossary.zh.md)

**How to use**: This glossary organizes the technical terms in the paper into eight domains. Each entry provides three layers of information: (1) what the concept is (definition and intuition); (2) where it appears in the paper and what role it plays; (3) how it relates to other entries. Readers from different backgrounds (music / machine learning / HCI) can read only the domains unfamiliar to them. Entry numbers (e.g., 3.7) are used for cross-references.

**Contents**

1. [Core concepts introduced by the paper (design-framework terms)](#1-core-concepts-introduced-by-the-paper-design-framework-terms)
2. [Electronic-music culture and aesthetics](#2-electronic-music-culture-and-aesthetics)
3. [Machine learning and neural audio synthesis](#3-machine-learning-and-neural-audio-synthesis)
4. [Statistics and empirical research methods](#4-statistics-and-empirical-research-methods)
5. [Human–computer interaction (HCI) and design research](#5-humancomputer-interaction-hci-and-design-research)
6. [Philosophy, sociology, and critical theory](#6-philosophy-sociology-and-critical-theory)
7. [System engineering and implementation](#7-system-engineering-and-implementation)
8. [Licensing, ethics, and scholarly publishing](#8-licensing-ethics-and-scholarly-publishing)

---

## 1. Core concepts introduced by the paper (design-framework terms)

**1.1 Organic operations**
The paper's overall framework: it reorganizes human interaction with neural latent spaces from "directly exposed coordinates" into three paradigms borrowed from botanical metaphors (Graft / Hybridize / Cultivate). The core claim is that the interaction entry point should provide a *semantic mediation layer*, so that producers can operate with the metaphorical language they already have ("wetter," "branching," "alive") instead of learning the machine's coordinate language. Appears in the title, abstract, §1, §4.3, and §5. The naming is grounded in the §3.7 community discourse analysis: the three paradigms directly correspond to the three highest-frequency metaphor clusters in Botanica community texts (growth/evolution 32%, ecology/symbiosis 24%, craft/cultivation 18%).

**1.2 Graft**
The first paradigm, after plant grafting: take two parent latent vectors z_a and z_b and compute a linear interpolation z_g = clip_3σ(α·z_a + (1−α)·z_b) with mixing coefficient α (Eq. 2, §4.3). The design intuition is producers' "A/B auditioning" habit — "70% kick drum plus 30% bass." It has the highest controllability and predictability, and the best learnability, of the three paradigms (§6.4: overall SUS 78.3 and learnability subscale 75.5, both the highest). Two points require careful reading: it is a Euclidean linear interpolation along the segment between the parents, **not** a geodesic on the manifold (see 3.10); and the 3σ clipping only prevents out-of-bound extrapolation — it does not guarantee that results lie on the effective support of the training distribution (see 3.11).

**1.3 Hybridize**
The second paradigm, after hybridization: the input is a semantic condition vector c (a CLAP text embedding, e.g., "wet," "grainy"); a conditional VAE produces the conditional posterior μ_ϕ(c), σ_ϕ(c), from which the system samples z_h = μ_ϕ(c) + σ_ϕ(c)⊙ε (Eq. 3). In essence: "intention-guided variation around a semantic point" — the user specifies the direction, the random term provides variation. The experimental result is counter-intuitive: Hybridize scores lowest on all five subjective dimensions, including experienced controllability (§6.4, §7.4) — conceptually the user controls c, but the sampling noise severs the "my input determines the output" causal chain.

**1.4 Cultivate**
The third paradigm, after cultivation: the only explicitly multi-step operation. The system maintains an interaction history H_t = {(z_i, a_i, r_i)}, estimates a reward function by GP regression, and at each step updates along the gradient of the predicted mean with a Thompson-style exploration term added (Eq. 4 / Algorithm 1). The producer cannot predict the single-step outcome, but can see the "growth direction" of the trajectory — trading controllability for long-term agency (§5.4). In the study, Cultivate scored highest on agency (5.9/7) and second-lowest on controllability (4.0); the failure case P5 ("the algorithm is stealing my judgment," §8.2) shows that the paradigm is not for everyone.

**1.5 Geometric disorientation**
A term coined by the paper for the producer's failure mode in front of a latent space, with an operational definition in §1: without repeated trial-and-error sampling, one cannot predict "what timbral change will result from moving Δz in some direction from the current z." Its measurable corollary is the "directed search success rate" (the proportion of attempts that reach a target timbre cluster within a limited number of steps); systematic collection of this metric is left to future work (§8.3). It is the starting point of the whole paper: the solution is not to teach producers the geometry, but to give the space a metaphorical language.

**1.6 Aesthetic-boundary quadrant**
A conceptual framework proposed in §7.4, dividing the design space of AI timbre tools by controllability (horizontal axis) and predictability (vertical axis). Graft sits top-right (high control / high predictability), Hybridize middle-lower-right (medium control / low predictability), Cultivate bottom-left (low control / medium predictability). The top-left quadrant (low control / high predictability) **is not occupied by any of the three paradigms**; its occupants are reference forms outside the framework — "black-box but behaviorally predictable" pure text-to-audio generation (AudioLDM, marked in gray in Figure 8). A key companion discussion: conceptual placement and participants' experienced ratings do not coincide (Hybridize is the counter-example, §7.4), and the paper argues the two should be measured separately. §7.10 concedes the framework is "a thinking scaffold, not a taxonomy" — the choice of axes is itself a design judgment.

**1.7 Semantic layer / semantic mediation layer**
The abstraction layer between "raw latent coordinates" and "user intent." Table 1 uses "semantic-layer abstraction" as the first comparison dimension for five existing tools: NSynth offers 16 sliders (no semantics), RAVE Studio offers bare coordinates (no semantics), AudioLDM offers text (semantics but no real-time intervention), and Botanica-OS offers organic-metaphor operations. From this comes the first design recommendation in §7.6, "semantic layer first," with a caveat: the semantic layer improves task-level experience (affordance, agency) but does not necessarily raise global usability scores — tool evaluation should distinguish the two.

**1.8 Bare-coordinate baseline**
RAVE-Studio-style interaction: every dimension of z is exposed directly for editing. It is the paper's foil (the B1 baseline in §6.8 and H1's comparison target) and the target of the criticism that "the interaction abstraction layer does not match producers' metaphorical language" (§2.5). Key experimental conclusion: with the same VAE backend (same d=16, same training data, same weights), changing only the interaction layer raises affordance from 2.8 to 5.4 (Wilcoxon p=0.031, exploratory) — the bottleneck is the interaction design, not the model (§7.9).

**1.9 Aesthetic drift**
Qualitative theme T3 (§6.5, 9/12 participants): over long Cultivate runs, the "aesthetic direction" of the timbre drifts with the trajectory — "I thought I was looking for a wet pad, and 20 steps later it had become some kind of bone-clicking sound, but the surprise pleased me" (P1). The paper reads this as "emergent redefinition of the aesthetic goal," and carries the concept forward in H3 (the criterion shifting from "precisely matching a preset" to "remaining interesting while evolving") and in the longitudinal study planned as future work (§8.3).

**1.10 The agency paradox (T2)**
A qualitative theme (8/12 participants): controllability falls while agency rises — "I don't know what the next step will produce, but it feels like *I* am steering it" (P9). It mirrors the conceptual/experienced gap of §7.4 and supports the paper's core argument: agency need not come from control; it can come from a process of "colluding with the algorithm" (after Eldridge & Bown; see 5.9).

**1.11 Algorithmic ingredient label**
An ethical mitigation design proposed in §7.7, analogous to a food ingredient label: AI tools should explicitly disclose the "ingredients" of their guidance mechanism — for example, "this guidance direction was trained on 120 ratings from 8 producers of the 2024 Chinese Botanica community" — so that producers can make an informed choice about accepting the algorithm's suggestions. It is a concrete response to the invisibility of algorithmic authority (see 6.1, 6.3).

**1.12 Hybrid workflow**
Three combinations of the paradigms recorded in §5.7 (N=3, exploratory observation): graft-then-cultivate, hybridize-seeding-plus-graft-pruning, and cultivate-led-with-graft-interventions. The paradigms can be combined, and producers mix them flexibly by situation; the paper accordingly recommends that future tools support switching between paradigms and preserve a cross-paradigm operation history. Figure 4 shows such mixed use through the eight stages of one participant's 90-minute session.

---

## 2. Electronic-music culture and aesthetics

**2.1 IDM (Intelligent Dance Music)**
A branch of 1990s British electronic music (key figures: Aphex Twin, Autechre, Squarepusher), characterized by "complex rhythm + micro-level timbre evolution + rejection of analog sound" (Reynolds calls it the "algorithmic aesthetic," §2.4). It is listening music rather than dance-floor music, and timbre design is itself compositional — which is why the paper chose it as the study object: IDM producers' implicit judgments about timbre are the most refined, and they are the most likely to experience "geometric disorientation" (1.5) in front of a latent space.

**2.2 Botanica (subgenre)**
The paper's study object: an IDM-derived subgenre that emerged around 2020 in the Bandcamp and independent-label ecosystem, an "ecological turn" taken by the IDM community after reflection on "algorithmic maximalism" (§3.1, §3.5; the §3.7 community discourse analysis is consistent with this). Its timbre fingerprint has three points (see 2.11): controllable but non-uniform grain density; slow-drifting, unstable harmonics; and describing timbre with biological-structure metaphors (growth/branching/hybridization). Its cultural context co-constructs with the circulation of Merlin Sheldrake's *Entangled Life* (a popular mycology book) in producer communities. Note: this is a very small (as the paper itself states) subgenre classification, and the ecological validity of the sample and conclusions is bounded by it (§8.1).

**2.3 Timbre**
A core concept of musical acoustics: the perceptual attribute that distinguishes two sounds of the same pitch and loudness ("is this a piano or a violin?"). Determined jointly by spectral envelope, onset/decay transients, and harmonic instability. The paper's object of study is *timbre design*, not composition or arrangement; Wessel's "timbre space" (§2.3; 5.8) is the classic proposal to embed multi-dimensional timbre perception into interaction mapping — latent-space tools are in a sense the neural incarnation of timbre space.

**2.4 Embodied craft**
§1 and §3.2's characterization of how IDM producers work: approaching an aesthetic goal that "cannot be fully described in words" by reading spectra, patching modulation chains, and auditioning take after take — knowledge stored in the body rather than in language. The latent space breaks this because parameter intuitions ("push the highs a bit") have no counterpart on a statistical manifold. This is the connective point between the paper's motivation and the §7.5 somaesthetics discussion (5.5).

**2.5 Granular synthesis**
A synthesis method with its theoretical foundation in Curtis Roads' *Microsound*: cut audio into 1–100 ms "grains," then reorganize them with controllable density, position, and envelope. The grain time scale sits exactly between the note and the sample, and it is the material basis of Botanica's timbre fingerprint (non-stationary density fluctuation) (§2.4, §3.1). The "grain density" parameter in the paper's interface comes from this tradition.

**2.6 The aesthetics of failure**
Proposed by Kim Cascone in 2000 (§2.4, §7.8): in the glitch tradition, digital errors (clipping, quantization noise, compression artifacts) are not technical failures but an acknowledgment and aestheticization of the medium's materiality. The paper uses it to argue the contestability of the "organic/inorganic" binary — some producers hold that the "inorganic" (algorithmic, glitch) is IDM's true tradition and Botanica is only one of its branches.

**2.7 Musique concrète and synthétisme**
Two electronic-aesthetic traditions distinguished in Demers' *Listening Through the Noise* (§2.4, §3.5): musique concrète "points sound at the external world" (recorded real-world sound as material), while synthétisme "points sound at its own materiality" (sound refers only to itself). The paper positions Botanica as a variant of synthétisme: it accepts the materiality of sound but describes it with ecological metaphors — timbre is "grown," not "parameterized" (§3.5).

**2.8 Generative music**
The tradition founded by Brian Eno (§7.11): design a rule system and let the system produce endless variations; the producer's agency lies in "designing rules" rather than performing, a posture of "set-and-forget." The paper positions Cultivate as the contrasting path: the producer remains present, continuously steering through explicit feedback, with agency migrating from "designing rules" to "designing guidance." The two postures form two paths of generative music — rule-driven off-line generation, and feedback-driven present guidance (§7.11).

**2.9 Ambient / dub techno / Techno**
The reference frame §3.6 uses to argue that the paradigms do not transfer across styles unchanged: Ambient (the Eno tradition) values growth but has long sustains and low density, and may not need Cultivate's trajectory evolution; dub techno uses "wet/steamy" metaphors but focuses on rhythm–timbre coupling; Techno needs the identity of a single sound and may not need Graft's dual parents. The §6.12 cross-style pilot (2 Ambient + 2 Techno producers, N=4, exploratory) tested the direction of these predictions.

**2.10 A/B auditioning**
A producer's daily habit: repeatedly alternating between two timbres for comparison. It is the source of DR2 (dual-parent support, §3.4) and the cognitive basis of Graft's dual-parent design — participants directly described graft results as "70% kick plus 30% bass" (§5.2), showing the interaction abstraction successfully hooked into an existing habit.

**2.11 Timbre fingerprint**
§3.1: the set of identifiable timbral features of a subgenre. The paper gives Botanica a three-point fingerprint (see 2.2); it serves to define the timbre space the training data must cover, and the criterion for "having Botanica aesthetic characteristics" in task design (the 8 target timbre clusters of §6.3, released with the repository under supplementary/target-timbres/).

**2.12 Liner notes / Bandcamp / Freesound / NSynth**
Proper nouns for research materials and data sources. Liner notes: album-sleeve texts, one corpus of the §3.7 discourse analysis. Bandcamp: the independent-music distribution platform (purchase ≠ training authorization; §4.5 has a dedicated licensing discussion — of the 12 artists involved, 11 granted written permission, and the remaining non-responding artist's works were removed); Bandcamp Daily is its official review outlet. Freesound: the CC-licensed environmental-sound library (the paper used 4 hours of birdsong, water, wind, and insects, with per-item CC0/CC-BY/CC-BY-NC records). NSynth: Google's 2017 note-level neural-synthesis dataset (12 hours used in §4.5, CC-BY 4.0).

**2.13 Artists in the lineage (Figure 7)**
Aphex Twin (the 1990s granular-aesthetics starting point of IDM), Autechre (2000s algorithmic maximalism), Oneohtrix Point Never (2010s surrealism), SOPHIE (the materiality turn — pushing the synthesizer's "matter" itself to the foreground), Boards of Canada, Arca. Figure 7 uses this lineage to argue the historiographical narrative that "agency migrates from the single human to human+machine," with Botanica-OS self-placed at the far right of the lineage (§7.1).

---

## 3. Machine learning and neural audio synthesis

**3.1 Neural latent space**
The low-dimensional representation space into which deep generative models compress high-dimensional data (audio waveforms): each point z encodes a "possible timbre," and the decoder maps the point back to a signal. Key properties: the space is statistical (shaped by the training-data distribution), non-Euclidean (perceptual distance ≠ Euclidean distance), and continuous (neighboring points decode to similar results, without a guarantee of semantic interpretability). The entire design of the paper revolves around one fact: this space is a black box to producers (§1).

**3.2 Variational autoencoder (VAE)**
Kingma & Welling 2014 (§2.2, §4.2). Consists of an encoder E_φ (mapping input x to distribution parameters μ_φ(x), σ_φ(x)) and a decoder D_θ. The difference from an ordinary autoencoder: the encoder outputs a probability distribution (the posterior q_φ(z|x)) rather than a point, and training constrains it with a two-term loss — a reconstruction term (the decoder must reconstruct the input) plus a KL regularizer (pulling the posterior toward the standard normal N(0,I)). The benefit of the KL term: the z space is "filled in," so any sampled z decodes to a plausible sample, without large holes. The paper's loss is Eq. (1), with β=0.5 weighting the two terms (the β-VAE convention).

**3.3 Reparameterization trick**
z = μ_φ(x) + σ_φ(x)⊙ε, ε∼N(0,I) (§4.2). The difficulty: training requires sampling z, but "sampling" is not differentiable. The solution moves the randomness to an external noise ε, making z a deterministic function of μ and σ — gradients can back-propagate through μ and σ. This is the core trick that makes VAEs trainable; the sampling in the paper's Hybridize operation (Eq. 3) takes the same form.

**3.4 KL divergence / KL regularization**
A measure of the difference between two probability distributions: D_KL(q‖p) ≥ 0, smaller means more similar, and it is asymmetric. In the VAE, D_KL(q_φ(z|x)‖N(0,I)) pulls every input's posterior toward the same prior — it is both the "anti-hole" regularizer and the knob controlling disentanglement strength in β-VAE (larger β, more regular space but blurrier reconstructions). The paper uses β=0.5 (Table A1).

**3.5 β-VAE and disentanglement**
Higgins et al. 2017 (§2.2, §4.2). Weighting the KL term with β>1 encourages each latent dimension to correspond independently to one semantic factor ("disentanglement") — the dimensions become interpretable. The paper's choice of a low-dimensional d=16 space shares this motivation: "so that producers can build intuition for every dimension" (§4.2), citing Esling et al.'s perceptual-regularization research showing that semantic-cluster separability in low-dimensional spaces can be explicitly preserved (Appendix A). The cost is reconstruction accuracy: spectral distance 0.42 vs. 0.18 for RAVE at d=64 (§4.7).

**3.6 RAVE (Realtime Audio Variational autoencoder)**
Caillon & Esling 2021 (§2.1). The representative work bringing VAEs to real-time audio synthesis: multi-scale adversarial training (a discriminator replacing part of the reconstruction loss) plus heavily trimmed inference, with decoding latency <1 ms — currently the most practical neural-synthesizer backend. The paper uses it in two places: (1) as the backbone training recipe for Botanica-OS (compressed to d=16); (2) as the B1 baseline in §6.8 — directly exposing RAVE's z coordinates, compared against the main system with the **same backend** (same d=16, same training data, same weights), demonstrating that the difference comes from the interaction layer, not the model. Its successor RAVE v2 (the IRCAM ACIDS toolchain, github.com/acids-ircam/rave) improves training stability and streaming inference; this study used v1, and v2's differences were not separately evaluated (§2.1).

**3.7 NSynth and the WaveNet autoencoder**
Engel et al. 2017 (§2.1). The NSynth dataset: ~300,000 four-second note recordings with acoustic annotations. Its WaveNet autoencoder uses WaveNet (an autoregressive audio model with dilated causal convolutions) for encoding and decoding, showing that neural networks can learn timbre diversity far beyond traditional wavetables — but its inference speed (seconds to synthesize one second of audio) confines it to being a "sample library" rather than a "real-time instrument." The paper treats it as the starting point of the "heavy generation, light interaction" lineage and borrows its data (§4.5).

**3.8 DDSP (Differentiable Digital Signal Processing)**
Engel et al. 2019 (§2.1, §4.4). Core idea: write classic DSP synthesizers (harmonics + noise) as differentiable operators so that a neural network predicts synthesis parameters rather than waveforms — small models, real-time capable, with "physical structure" built into the generative process. The paper's backend is the DDSP harmonic-plus-noise synthesizer: 64 harmonic oscillators (amplitudes h_k, fundamental f₀) + filtered noise (amplitude n(t), time-varying filter coefficients b(t)), with the synthesis formula given as Eq. (5). DDSP-violin is its instrumental application (one of the compared tools in Table 1).

**3.9 Harmonic-plus-noise model**
The classic speech/instrument synthesis model: signal = harmonic part (a steady sinusoid series at integer multiples of f₀, determining pitch and harmonic timbre) + noise part (a random component with time-varying spectral shape, determining breathiness, bow noise, and other "life"). The four parameter groups of §4.4 (f₀, h_k, n, b) are its parameterization. Producer-friendly point: these parameters map directly onto traditional synthesizer concepts.

**3.10 Geodesic vs. linear interpolation**
A geodesic is the "shortest path along the surface" between two points on a curved surface/manifold (like a great-circle route on Earth). The paper's Graft, Eq. (2), is a Euclidean linear interpolation and **not** a geodesic: under the VAE's Euclidean parameterization, the midpoint of the segment can fall into low-density regions (audibly "hollow" timbres), which is not the same as a perceptually "natural transition" — the paper's interpolation-path density audit found 23% of interpolation midpoints falling below the 5th percentile of training density (§4.3). "Truly manifold-aware interpolation" is listed as future work (§8.3). This is a concrete manifestation of the §7.2 epistemological paradox (6.5): straight geometrically, not straight perceptually.

**3.11 3σ clipping (clip to 3σ boundary)**
clip_3σ(·): clipping every dimension of the interpolation/update result to within ±3 standard deviations of that dimension's training-set mean. Statistical intuition: 99.7% of a normal distribution's mass lies within ±3σ; anything beyond is almost certainly a region the training distribution never covered. Purpose: to keep operations from pushing z into never-trained regions and producing garbage. Limitation (explicit in §4.3): it is a per-dimension box constraint — a necessary but not sufficient condition; low-density holes can still exist inside the box (see 3.10).

**3.12 AudioLDM (Audio Latent Diffusion)**
Liu et al. 2023 (§2.1). It brought latent diffusion to text-to-audio: first train a mel-spectrogram VAE, then train a diffusion model in latent space conditioned on CLAP text embeddings. High quality, but generating one clip takes 5–30 s, with no trajectory and no real-time intervention. Uses in the paper: a comparison target in Table 1; the B2 baseline in §6.8 (qualitative reference only — the task form differs fundamentally, so it does not enter formal statistical inference); and the reference form occupying the "low-control / high-predictability" quadrant of §7.4's four quadrants (gray marker in Figure 8).

**3.13 Diffusion models and denoising trajectories**
A generative paradigm: during training the data is progressively noised and the process's reversal is learned; at generation, the model denoises from pure noise in multiple steps. DiffWave (Kong et al.) is its audio version (§2.6). §2.6 notes that the "denoising trajectory" is conceptually isomorphic to Cultivate's "trajectory interaction" — both are time-extended generative processes — but current tools black-box the trajectory and expose only the final result. This is one source of Cultivate's design inspiration.

**3.14 Conditional VAE (cVAE)**
A VAE variant with an added condition input: the decoder (and/or encoder) additionally receives a condition c, making p(z|c) controllable. The paper's Hybridize branch (§4.3): CLAP text embedding → 2-layer MLP (hidden 128, GELU) → (μ_ϕ(c), σ_ϕ(c)), jointly trained with the main VAE and sharing the decoder, trained on 3,214 audio–text pairs (9:1 train/validation split). Its failure mode: when c is an out-of-distribution concept ("flame-like wetness"), σ_ϕ(c) inflates and samples leave the distribution's support (§8.2) — which motivated the OOD detection and fallback mechanism (3.21).

**3.15 CLAP (Contrastive Language-Audio Pretraining)**
Wu et al., ICASSP 2023 (LAION-CLAP). Two-tower contrastive learning: audio and text encoders map both modalities into one embedding space, pulling paired audio/text together and unpaired apart. Effect: any natural-language phrase becomes a vector c in the space, with semantically similar phrases lying close together. The paper uses it as the conditioning source for Hybridize (laion/clap-htsat-fused weights, unmodified, §4.3, Appendix B; code MIT, weights CC-BY 4.0 — see the repository's THIRD_PARTY_NOTICES). Limitation (§4.7): CLAP was not trained on Botanica metaphors, so terms like "mycelial" or "leaf-vein" take trial and error to land on a suitable c.

**3.16 UMAP (Uniform Manifold Approximation and Projection)**
A nonlinear dimensionality-reduction/manifold-visualization method (§4.6, Figures 3/6): find a low-dimensional embedding of high-dimensional data that preserves local neighborhood structure and topology. The paper uses it to project the d=16 latent space to a 2D map so producers can see "where they are" (DR4, visible trajectories). Two hyperparameters: 50 neighbors (the local/global structure trade-off) and min_dist 0.1 (crowding between points), in Table A1. Incrementally updated every 100 interactions (§4.6). Note: distances in the UMAP projection are a visualization approximation, not the latent space's true metric — which itself participates in the §7.2 epistemological mismatch (6.5).

**3.17 Gaussian-process regression (GP) and the Matérn 5/2 kernel**
Bayesian non-parametric regression (§4.3, Algorithm 1): treat the "reward function" as a random function; given observed points (z_i, r_i), the reward at any new point z follows a posterior distribution — with a predicted mean μ̂(z;H) (the most likely reward) and predicted standard deviation σ̂(z;H) (the model's uncertainty). The kernel defines how reward correlation decays with distance: Matérn 5/2 (Table A1) is slightly "rougher" and twice differentiable compared to the squared-exponential kernel, a common Bayesian-optimization default. Cost O(t²)–O(t³): this is why the history is truncated to the most recent 100 interactions (§4.3, §4.7). A GP has no "order" to speak of; the paper approximates the gradient of the GP mean by first-order finite differences (Algorithm 1, line 2).

**3.18 Multi-armed bandit / exploration–exploitation**
A classic reinforcement-learning framework (§4.3): repeatedly choosing actions for rewards in an uncertain environment requires balancing "exploiting known-good options" and "exploring possibly better ones." Cultivate models "taking the next step in z space" as a bandit problem: the history H_t is the observations, the GP is the belief model.

**3.19 Thompson sampling**
An exploration strategy: act randomly according to "current beliefs" — regions of high uncertainty are more likely to be explored, and the behavior converges to exploitation once uncertainty shrinks. The paper's implementation is an approximate form: adding λ·σ̂(z_t)⊙ε to the update (larger σ̂ → more random steps), with λ=0.3 as the exploration coefficient (Eq. 4, Algorithm 1, Table A1). Participants' "aesthetics of surprise" experience (T3, 1.9) is the direct consequence of this term in high-uncertainty regions — the paper's most direct mapping from "algorithm mechanism → user experience" (§5.6).

**3.20 Gradient ascent on the reward surface**
The paper writes Cultivate's update as z_{t+1} = z_t + η·∇_z μ̂ + λ·σ̂⊙ε (Eq. 4) — climbing the estimated reward surface, with η=0.05 as the learning rate. Eq. (4) and Algorithm 1 are consistent: the gradient term is exploitation; the σ̂-weighted random term is exploration (§5.6).

**3.21 Out-of-distribution (OOD) and support**
Support: the region where a probability distribution "actually takes values." OOD: the input falls outside the training distribution's support — the model's outputs for it carry no guarantee. Two places in the paper: (1) Hybridize's OOD condition detection — if any component of σ_ϕ(c) exceeds the 99th percentile of the training set's conditional-posterior standard deviations, the condition is judged OOD and the system falls back to the nearest valid condition in embedding space with a user-facing notice (§4.3); this mechanism is a post-hoc engineering iteration triggered by a failure case and was **not part of the evaluated version** (§8.1, version deviation); (2) Graft's 3σ clipping is only a coarse-grained OOD defense (see 3.11).

**3.22 CLIP / StyleGAN / StyleCLIP / interFaceGAN (image-domain counterparts)**
The image-domain latent-tool family used for comparison in §2.7 and §4.8. StyleGAN (Karras et al. 2019): an architecture controlling generation layer-by-layer with style vectors; its W space supports editing. interFaceGAN (Shen et al. 2020): trains supervised "latent directions" w; editing is z←z+αw. StyleCLIP (Patashnik et al. 2021): turns natural language into StyleGAN editing directions via CLIP's image–text alignment. The paper uses them to argue: the image domain already has mature "semantic latent editing," while audio remains at bare coordinates — and audio's real-time, temporal dimension makes "Cultivate-style" interaction even more important in audio.

**3.23 MusicGen / AudioGen / EnCodec / AudioCraft (the newer text-conditioned tools)**
The circa-2023 literature of §2.1. EnCodec (Défossez et al. 2023): an RVQ (residual vector quantization) neural codec that compresses audio into discrete tokens — the foundation for large-scale audio models. AudioGen (Kreuk et al.): an autoregressive LM for text-to-sound-effects. MusicGen (Copet et al. 2023): conditional-LM music generation over EnCodec tokens with melodic conditioning; AudioCraft is Meta's unified toolchain for them. Common point: the prompt–generate "finished artifact" paradigm, complementary to this paper's "real-time operable latent state" paradigm (§2.1).

**3.24 VampNet and masked acoustic token modeling**
Flores García et al. (§2.1): convert audio into discrete tokens and model them "cloze-style" — randomly mask some tokens and have the model iteratively complete them in parallel. Naturally supports inpainting/variation (generating variants from a motif). The paper cites it to establish the feasibility of "completion/variation" interaction; the difference from Cultivate is that VampNet operates on discrete tokens, not real-time continuous trajectories.

**3.25 Stable Audio**
Evans et al., 2024 (§2.1): latent diffusion with timing-conditioning embeddings, generating up to 95 seconds with second-scale rendering. It represents the state of "generation quality and duration," but remains a prompt→artifact non-real-time paradigm (the paper uses it as the reference form of the fourth quadrant in §7.4).

**3.26 CNN / LSTM / MLP / GELU**
Network components used in the paper (§4.5, Table A1). CNN (convolutional network): extracts local time–frequency features; 4 layers in the encoder. LSTM (long short-term memory): a recurrent network for sequential dependencies; 2 layers in the encoder. MLP (multi-layer perceptron): fully connected feed-forward network; 3 layers in the decoder, 2 in the conditional branch. GELU: a smooth activation function (a "gate" softer than ReLU), used in the conditional branch.

**3.27 Adam / batch size / epoch / learning rate / cross-validation**
Training terms (§4.5, Table A1). Adam: the adaptive-momentum optimizer, learning rate 1×10⁻⁴. Batch size 32: 32 samples per gradient update. 200 epochs: 200 passes over the full training set (~18 hours on a single A100). Cross-validation (5-fold): split data into 5 parts, holding each out in turn to prevent hyperparameters from "memorizing answers" — the paper uses it to select model hyperparameters (targeting validation-set reconstruction loss) and Cultivate's η, λ (grid search, §5.6).

**3.28 Spectral distance**
An objective reconstruction/generation-quality metric (§4.7): the average distance between original and reconstructed spectra. The paper uses it to quantify the cost of d=16: 0.42 (Botanica-OS) vs. 0.18 (RAVE at d=64) — lower is better. This accuracy gap caps output quality, but because the RAVE baseline uses the same d=16 backend, it does not affect the internal validity of the "interaction layer vs. backend" comparison (§4.7, §6.8); sound-quality satisfaction was not collected separately and is listed as a measurement gap (§8.1).

**3.29 FAD (Fréchet Audio Distance)**
An objective metric listed as future work in §8.3: the audio version of FID — extract embeddings with a pretrained network (VGGish-type) and compare the embedding distributions of generated and real sets with the Fréchet distance. Closer than pairwise spectral distance to "distributional perceptual similarity," suited to evaluating generative models as a whole rather than single samples.

**3.30 Shannon entropy and access entropy**
The information-theoretic measure H = −Σ p_i·log₂ p_i (bits): the more dispersed the distribution, the higher the entropy (§6.7). The paper maps participants' z-access trajectories onto the UMAP grid and computes visit frequencies p_i: Graft entropy 1.8 bits (compact), Hybridize 2.6 (high-entropy trial-and-error), Cultivate 3.4 (extended evolution). Note the paper's framing: this is descriptive statistics with no significance claims; the validity of trajectory entropy as a candidate "organicity" metric remains to be tested (§8.4).

**3.31 Fractal dimension / self-similarity / ecological-psychology indicators**
Candidates for an objective "organicity" metric among the open questions of §8.4: fractal dimension (morphological complexity across scales; plants/coastlines 1<D<2), self-similarity (statistical similarity across time scales), and ecological-psychology indicators (the Gibsonian tradition: the perceptual affordances of the animal–environment relation). None has yet been validated in latent tools.

**3.32 Mel spectrogram**
The audio front end of §4.1: frame the waveform, apply the Fourier transform to obtain spectra, then bend the frequency axis to the Mel scale (modeling human pitch perception's nonlinearity: fine resolution at low frequencies, coarse at high frequencies) and take energies. It is the standard input representation for neural audio models.

---

## 4. Statistics and empirical research methods

**4.1 Mixed methods**
A research design combining quantitative (questionnaires, logs) and qualitative (interviews) data (§6.1). The paper's combination: SUS/Likert questionnaires + operation-log trajectory analysis + semi-structured interview thematic analysis. They complement each other: numbers answer "by how much," interviews answer "why and how."

**4.2 Within-subject design**
Every participant experiences all conditions (all three paradigms) rather than being assigned to one group (between-subjects). Advantage: each person is their own control, individual differences cancel out, and small samples retain power; disadvantage: order/fatigue/practice effects (see 4.5), which balanced designs must counteract. N=12 chose a within-subject design precisely for this reason (§6.1).

**4.3 Latin-square balancing**
An order-balancing technique: make every "condition–position" combination occur equally often. Three conditions have six orders; the paper used a Latin-square rotation (GHC/HCG/CGH, 4 participants each, §6.1), so that each condition appears once at each of positions 1/2/3 — making order effects testable in the analysis (order_fatigue in reanalysis.py). The assignment design is released with the repository under supplementary/latin-square/.

**4.4 SUS (System Usability Scale)**
Brooke 1996's 10-item, 5-point usability scale: alternating positive/reverse items, scored on 0–100. Industry benchmarks: 68 is about "average," 70+ good, 80+ excellent. The paper reports two measures (§6.1): the overall SUS (all 10 items); and the **learnability subscale** — item 4, "I need the help of a technical person to be able to use this system," and item 10, "I needed to learn a lot of things before I could get going with this system" (both reverse-worded), reverse-scored and separately rescaled to 0–100. The abstract's 75.5 is Graft's learnability subscale (the overall SUS is 78.3 — the two are not the same thing). One caveat: the subscale contains only two items and received no reliability check (§6.1); treat its score as a reference indicator, interpreted jointly with the overall SUS and the interview evidence. The original questionnaires are in the repository at study/questionnaires.md.

**4.5 Order / fatigue / practice effects**
The three threats to within-subject designs (§6.9, §6.11): order effects — a condition's score depends on its position; fatigue — attention declines in the third of three 20-minute tasks (P11's report); practice — improving with repetition. The paper's countermeasures: Latin-square balancing plus statistical tests — a Kruskal-Wallis trend across positions (H(2)=1.4, p=0.50) and a first-vs-last position comparison (Wilcoxon p=0.38), both non-significant (§6.11).

**4.6 Likert scale (7-point)**
Rensis Likert's summated-rating scale: choose 1 (strongly disagree) – 7 (strongly agree) for each statement. The paper uses five dimensions with three items each: affordance, agency, controllability, pleasure, and aesthetic fit (construct definitions in §6.1; the complete original items are in Appendix D and in the repository at study/questionnaires.md). Analysis takes item means.

**4.7 Paired t-test, and why it was not used**
Compares the mean difference of the same participants under two conditions, assuming normally distributed differences. With N=12, normality is not guaranteed, so the paper's main analysis does not use t-tests, adopting the non-parametric framework instead (4.8–4.11); §6.11 states this choice explicitly.

**4.8 Wilcoxon signed-rank test**
A non-parametric test for paired data: rank the absolute differences within pairs and sum by sign; it requires no normality, only roughly symmetric difference distributions. The paper's main post-hoc pairwise test (with Holm correction); the baseline comparisons (N=6) used the exact version (Wilcoxon exact, e.g., affordance p=0.031).

**4.9 Friedman test**
The within-subject multi-condition analogue of Kruskal-Wallis: rank each participant's scores across the k conditions, then test whether the rank structure is random (χ² approximation). The paper's main analysis (per metric across the three paradigms, §6.4); Friedman requires complete blocks, so it is based on the N=11 participants with complete data in all three conditions; post-hoc pairwise Wilcoxon tests follow only when significant.

**4.10 Post-hoc tests and the multiple-comparisons problem**
One test has a 5% false-positive rate (α=0.05); with 3 pairs × 7 metrics = 21 tests, the probability of at least one false positive rises to about 66% — the multiple-comparisons problem. Correction is mandatory. The paper reports all pairwise comparisons (no cherry-picking).

**4.11 Holm-Bonferroni correction**
A step-down procedure: order the m p-values from smallest to largest, multiply successively by m, m−1, …, 1 with cumulative-maximum enforcement, controlling the family-wise error rate (FWER) — the probability of at least one false rejection among all true nulls ≤ α. More powerful than plain Bonferroni (multiply all by m); it is the paper's scheme (the holm function in reanalysis.py).

**4.12 Effect sizes: Cliff's δ and Cohen's d_z**
p-values answer "is there a difference"; effect sizes answer "how large" — mandatory reporting in small samples. Cliff's δ: for a randomly drawn pair, (P(X>Y) − P(X<Y)), ranging −1..1, readable as the probability advantage of X over Y; common benchmarks |δ|≈0.15/0.33/0.47 for small/medium/large (the paper's Graft vs. Hybridize δ=0.61 is a large effect). Cohen's d_z: the standardized mean difference for paired designs = mean of differences / SD of differences (the H1 baseline comparison gave dz=0.03).

**4.13 Bootstrap confidence intervals (95% CI)**
Resample the sample with replacement 10,000 times, compute the mean each time, and take the 2.5%/97.5% percentiles as the 95% CI (§6.4). The paper's Graft [74.1, 82.5] and Hybridize [69.8, 76.9] overlap partially — note that overlap (or not) of two independent CIs is not itself a test; judgment defers to the paired test (§6.4).

**4.14 Bayes factor BF₁₀**
The ratio of data probability under two hypotheses, P(data|M1)/P(data|M0) (§6.11). BF₁₀=4.7 means the data are 4.7 times more likely under "there is a difference" than under "no difference." The paper's evidence grading (Jeffreys tradition, §6.11): BF₁₀ < 3 anecdotal (not evidence), 3–10 moderate, > 10 strong. Graft vs. Hybridize BF₁₀=4.7 is moderate evidence; Graft vs. Cultivate BF₁₀=1.2 constitutes no evidence. Computation uses the JZS Cauchy prior r=0.707 (implemented with pingouin; the script skips the column if unavailable).

**4.15 Linear mixed model (LMM)**
A regression containing both fixed effects (the factors of interest: condition, group, order) and random effects (sampling sources: participants' individual baselines, i.e., random intercepts) (§6.11): score ~ condition*group + order + (1|subject). Paper's results: a significant condition main effect (χ²(2)=7.1, p=0.029), with the condition×group interaction (p=0.54) and order position (p=0.61) non-significant — consistent with the main analysis. Advantages: no need for fully balanced designs, estimates several factors' contributions simultaneously, more robust to missingness. Implemented with statsmodels as a sensitivity analysis.

**4.16 Statistical power**
1−β: the probability of detecting an effect when it truly exists. §8.1 (limitation 1) gives a quantitative estimate: at α=0.05 (two-sided), a within-subject paired test with N=12 requires an effect of dz≈0.89 to reach 0.80 power; a single test has only about 35% power for a medium effect (dz≈0.5) and about 18% for dz≈0.33 — even the large effect observed for Graft vs. Hybridize (δ=0.61) had only about a 49% chance of detection in this sample. "Not significant" must therefore not be read as "no difference." Subgroup comparisons with 6 per cell have even lower power — the methodological basis for labeling all of §6.6 "exploratory."

**4.17 Mann-Whitney U / Kruskal-Wallis**
Rank tests for two independent samples (Mann-Whitney U, i.e., Wilcoxon rank-sum) and for multiple independent samples (Kruskal-Wallis). Uses in the paper: the former for the producers-vs-researchers subgroup exploration (§6.6); the latter for the order-position trend check (§6.11).

**4.18 IQR (interquartile range) and median**
Median: the middle value after sorting (robust to skew and outliers, suitable for right-skewed quantities like trajectory length). IQR: the 75th percentile minus the 25th — the span of the middle 50% of the data. §6.7's "Graft median trajectory 7.2 steps (IQR 5–10)" reads: half the participants' trajectory lengths fell between 5 and 10 steps.

**4.19 SD vs. SE**
The SD describes dispersion within the sample (variation between individuals); SE = SD/√N describes the estimation uncertainty of the sample mean and is always smaller. The paper's Table 7 reports M±SD, while Figure 5a's error bars are SE (the caption distinguishes them to prevent confusion).

**4.20 ITT vs. per-protocol (intention-to-treat / per-protocol)**
Clinical-trial tradition (§6.4, §6.11): ITT analyzes everyone as originally assigned (including dropouts, with conservative missing-data handling), preserving the meaning of randomization; per-protocol analyzes only completers — closer to the "experienced reality" but biased. The paper's handling: the main analysis is per-protocol (Cultivate N=11, P5 withdrew), and an ITT sensitivity analysis with conservative imputation (filling missing values with the lowest observed score across the three paradigms) re-ran the main test with the same direction (Friedman χ²(2)=7.2, p=0.027).

**4.21 Missing data**
The handling principle for the 11/12 follow-up responses (§6.10): no imputation, descriptive reporting only, no responder/non-responder comparability test (meaningless at this N) — and the results' scope of application is explicitly "the responder subset."

**4.22 Descriptive statistics**
Statistics that characterize the sample itself without inference tests (means, medians, IQR, entropy). The paper explicitly frames §6.7 (trajectory analysis) and §6.10 (follow-up) as descriptive — distinguishing "description" from "inference" is key to reading this paper's conclusions at small samples.

**4.23 Task–paradigm confound**
The design confound flagged in §8.1 (limitation 2): the three paradigms were evaluated on three different tasks (Graft = find a satisfying intermediate timbre between two parents; Hybridize = generate three variants with semantic tags and pick the best; Cultivate = cultivate a target timbre within 20 steps; §6.3), whose difficulty, goals, and time structure are not equivalent. SUS and Likert ratings therefore measure the "paradigm + task" package, and between-paradigm differences cannot be attributed to the interaction paradigm alone; the study controlled neither task equivalence nor modeled task as a covariate. Separating the two requires task-equivalent comparison designs (§8.3, future work).

---

## 5. Human–computer interaction (HCI) and design research

**5.1 Research through Design (RtD)**
The research path in Frayling's classification of "producing knowledge by doing design" (§1.1, citing Frayling 1993, *Research in Art and Design*): building a prototype (Botanica-OS) is itself inquiry, and knowledge partly resides in the artifact. The risk is that design judgments carry bias — the paper's hedges: formalizing design requirements up front (DR1–DR5, Table 2), evaluating them post hoc with a user study, and critically reflecting on its own metaphors in §7.8.

**5.2 Affordance (also translated as 可供性/给养)**
Originated in Gibson's ecological psychology and brought into design by Norman's *The Design of Everyday Things* (1988) (§7.5): the action possibilities an object "offers" to its perceiver. Norman's version emphasizes "perceived action cues" — a door handle suggests pulling. The paper's first Likert dimension uses it to measure "the system showed me what I can do" (§6.1). Translation note: 示能性/可供性/给养 are three common Chinese renderings; the paper uses 示能性 throughout.

**5.3 Agency and aesthetic agency**
The sense of authorship — the experience that "the outcome's direction is guided by me" (the second Likert dimension). McCormack et al.'s analysis of autonomy, authenticity, authorship, and intent in computer-generated art is the paper's direct reference for agency (§1, §2.4); Eldridge & Bown's claim that "the experience of agency in algorithmic aesthetics depends on the maker's history of material perception" (§6.6) explains the subgroup difference. The paper's empirical contribution: agency and controllability are separable (Cultivate highest on agency at 5.9, second-lowest on controllability at 4.0).

**5.4 Direct manipulation and semantic distance**
Hutchins, Hollan & Norman 1985 (§7.5): direct-manipulation interfaces make users feel they are "operating the object itself" rather than issuing commands; the feeling of directness is characterized by two distances — semantic distance (how directly intention maps to action) and articulatory distance (how directly action maps to feedback). The paper classifies Graft as a tight mapping and Cultivate as a loose one.

**5.5 Somaesthetics**
Shusterman (§7.5): the philosophical program that places bodily perception and cultivation at the core of aesthetics. The paper uses it to interpret the Cultivate experience — an embodied loop (body–tool–environment) replacing direct control; the "like tending a plant" embodied feeling (P9).

**5.6 Designing constraints**
Magnusson (§2.3, §5.5, Computer Music Journal 2010): a good instrument's essence is not "letting users do anything" but "amplifying particular expression through the right constraints." The paper treats the three paradigms as three "constraint choices" — each rules out part of the possibility space in exchange for a different kind of agency. This is the pivotal concept in the paper's dialogue with the musical-HCI tradition.

**5.7 Digital lutherie and instrumentality**
Jordà's terms (§2.3; from his doctoral thesis *Digital Lutherie*): designing instruments for digital media the way a luthier designs stringed instruments. Instrumentality: the quality of a tool being treated as an "instrument" (practiceable, embodied, expressive) rather than a "tool" (used and discarded once the task is done). The paper criticizes existing neural-audio tools as "ML model + simple UI," lacking instrumentality.

**5.8 Timbre space**
Wessel 1979 ("Timbre Space as a Musical Control Structure," Computer Music Journal): embedding the multi-dimensional perceptual differences of timbre into a geometric space, with timbral transformation expressed as movement through the space — the intellectual ancestor of latent-space timbre tools. What the UMAP projection visualizes is exactly this kind of "timbre map."

**5.9 Agency in algorithmic aesthetics (Eldridge & Bown)**
Eldridge & Bown (§3.2, §6.5, §6.6; a chapter in *The Oxford Handbook of Algorithmic Music*): algorithms are not tools but "aesthetic actors," and the human–algorithm relationship is experiential and collusive. The paper draws on them twice: agency can come from collusion rather than control (T2, 1.10); and the explanation of the subgroup difference — producers treat the algorithm as a "thing," researchers remain wary (§6.6).

**5.10 Thematic analysis and open coding**
The standard qualitative method (§6.5, Appendix C): attach conceptual labels to transcript sentences (open coding) → group related labels (axial coding) → distill themes (T1–T4). Pipeline: Whisper large-v3 transcription → manual proofreading → coding; the coding manual is released with the repository under supplementary/coding-manual/. Declared limitations: coder = designer (confirmation-bias risk), no second coder, no member checking (§6.5, §8.1).

**5.11 Inter-coder reliability (Cohen's κ) and member checking**
The two quality checks of qualitative analysis: κ (kappa) — the agreement rate between two independent coders (corrected for chance agreement); member checking — feeding analytical conclusions back to participants for verification. The paper did neither (resource constraints) and states so as a limitation (§8.1).

**5.12 Confirmation bias and exploratory analysis**
Confirmation bias: the tendency to find data that supports one's own design. The paper's handling chain: §1.1 (RtD bias statement) → §2.5 (the interface critique is the author's own assessment) → §6.5 (coder bias) → §6.6/§6.8 (subgroups and baselines all labeled "exploratory") → §8.1 (limitation 4). Exploratory analysis means analysis "aimed at generating hypotheses, not verifying them"; its findings need replication in larger samples before counting as conclusions.

---

## 6. Philosophy, sociology, and critical theory

**6.1 Algorithmic authority and authorial migration**
§7.3, §7.7: when the algorithm "suggests" generative directions via a reward function (Cultivate) or a conditional posterior (Hybridize), judgment authority partially migrates from producer to algorithm — and the migration is covert (in the form of suggestion rather than decision). Participant P10's words are the best gloss: "If I accept this suggestion, is my own aesthetic judgment atrophying?" — theme T4 (6/12 participants; 5/6 in the ML-researcher subgroup).

**6.2 Cultural capital and distinction (Bourdieu)**
Bourdieu's *Distinction* (1984): aesthetic judgment is never neutral; it is the distribution and reproduction of cultural capital — "taste" marks class. §7.3/§7.7 use this to upgrade "the reward function encodes 8 producers' preferences" from a technical issue to one of social power: that group gains disproportionate aesthetic say through the algorithm, enforcing a covert aesthetic discipline (6.3).

**6.3 Aesthetic discipline**
§7.7's derived concept: the shaping of user taste by algorithmic guidance. The risk is invisibility — the user does not know where the direction came from. Countermeasures: the "algorithmic ingredient label" (1.11) plus bias audits (compute reward-prediction residuals grouped by rater and examine whether residuals cluster in UMAP space; the audit script is future work, §7.7, §8.3).

**6.4 Mechanical reproduction and aura (Benjamin)**
Benjamin 1936: technical reproduction strips the artwork of its unique "here and now" — the aura withers. §7.3 of the paper points to a different situation: the algorithm intervenes by "suggesting rather than deciding," the aura survives, but the origin of authorship becomes untraceable — not reproduction destroying aura, but algorithmic generation making "who is the author" undecidable.

**6.5 The epistemological paradox**
§7.2: describing latent-space operations with botanical metaphors involves an ontological mismatch — botanical grafting happens in three-dimensional Euclidean space (consistent with embodied perception), while latent-space "grafting" happens on a statistical manifold (geometrically straight ≠ perceptually natural; see 3.10). The paper's stance: acknowledge the paradox, treat the metaphor as "a communication tool, not a geometric description," and hold that the paradox should be explicitly acknowledged and exploited by designers rather than eliminated (§9). Participant P2 calling her most-used vector "a monster" is the empirical evidence of this mismatch.

**6.6 Autopoiesis (Maturana & Varela)**
Maturana & Varela 1980: living systems are "self-producing" — their components continuously produce the very organization that produces them, and the system thereby maintains an "identity." The paper's borrowing (§7.11): when the Cultivate system maintains a stable attractor in z space (its own "identity"), the producer's relation to it becomes closer to "coexisting with an autonomous system" than "using a tool."

**6.7 Colonial epistemology**
§7.8's self-critique: botanical metaphors carry the Enlightenment/colonial epistemology of "nature = governable growth" (classification, naming, domestication), and organic operations may inadvertently reproduce that tradition. This forms a deliberate tension with §3.5's "cultural-contextual grounding of the metaphor choice" — the paper's strategy is to acknowledge the metaphor's historical baggage and let the subgenre's cultural context decide "which metaphor."

**6.8 Information and meaning (Borgmann)**
Borgmann's *Holding On to Reality* (1999): the core challenge of the information age is information overload and its disconnect from meaning. The conclusion chapter uses it to close (§9): the organic-operations framework is one response to this challenge — the botanical metaphor is an embodied, cross-culturally shared language through which operations in high-dimensional statistical space regain access to producers' systems of meaning.

**6.9 Process as aesthetics (the generative-music special issue, Collins & Brown)**
The *Contemporary Music Review* generative-music special issue edited by Collins and Brown (2009; §2.4, §7.6, §7.11): aesthetic agency lies not in the "generated result" itself but in the "collaborative process between human and algorithm." The paper uses it twice: the "aestheticize the trajectory" design recommendation (§7.6) and Cultivate's "agency accumulates in the interaction history" (§7.11).

---

## 7. System engineering and implementation

**7.1 End-to-end latency and 47±3 ms**
The total time from "the input signal entering the audio driver" to "the corresponding output block appearing at the interface" (the measurement protocol of §4.4: loopback, 1,000 measurements, mean ± SD). 100 ms is the psychological threshold for "performative interaction" (an instrument must respond immediately); DR5 adopts it as the acceptance line. Table 4 decomposes the 47 ms: I/O buffering 8.0 + encoding 11.5 + operation layer 5.5 (incl. GP prediction) + decoding 9.0 + synthesis 7.5 + inter-process communication 5.5; model inference totals about 29 ms (the "CPU inference ~30 ms" of §4.6). The latency breakdown is released as results/table4_latency.csv in the repository.

**7.2 Audio buffering and block size (128 samples @ 16 kHz)**
Digital audio is processed in blocks: smaller blocks mean lower latency but more frequent CPU interrupts. At 16 kHz, a 128-sample block = 8 ms/block (128/16000). 16 kHz (not 44.1 kHz) serves real-time performance: 8 kHz bandwidth suffices for timbre design while more than halving computation.

**7.3 Sample rate**
The number of waveform samples per second. 16 kHz = 16,000 samples/s, Nyquist limit 8 kHz — covering most of the spectrum that matters for timbre design while losing high-frequency detail. Part of the §4.7 sound-quality limitation.

**7.4 WebSocket / FastAPI / React + TypeScript**
The system's communication and UI stack (§4.6). WebSocket: a persistent bidirectional connection between browser and server (unlike request–response HTTP), used to stream latent vectors and audio parameters in real time. FastAPI: Python's modern web-service framework hosting the model-inference backend. React + TypeScript: the frontend UI framework plus a type-safe language, rendering the UMAP map and operation cards.

**7.5 librosa**
The Python audio-analysis library (§4.6; McFee et al. 2015): mel spectrograms, feature extraction, resampling, etc. The paper uses it for front-end audio processing.

**7.6 Hardware platforms: MacBook Pro M2 / NVIDIA A100**
§4.6, Appendix B. The M2 (Apple-silicon laptop) is the deployment/study platform — CPU-only inference suffices (model inference ≈30 ms; see the 7.1 breakdown), demonstrating the system's "single-machine usability." The A100 (NVIDIA data-center GPU) is the training platform — 200 epochs ≈ 18 hours. Splitting the two is the typical research-prototype configuration: train once, infer countless times.

**7.7 PyTorch 2.1 / differentiable_dsp**
The implementation stack. PyTorch: the mainstream deep-learning framework (automatic differentiation is the prerequisite of differentiable DSP). differentiable_dsp: a Python reference implementation of DDSP providing differentiable harmonic-oscillator and filtered-noise operators.

**7.8 UMAP incremental updates and caching**
§4.6: the projection is precomputed at service startup (a one-time cost), then incrementally updated every 100 interactions — balancing "the map must track the user" against "recomputation is expensive." Such UI/visualization costs are also included in Table 4's latency decomposition (WebSocket serialization and UI sync, 5.5 ms).

**7.9 JSON serialization / loopback measurement**
JSON: the cross-process data format used by WebSocket payloads (serialization overhead counted in latency). Loopback: the standard practice for latency measurement — the signal enters at the physical input port and exits at the physical output port; measure the round trip and subtract known buffers to obtain the true end-to-end latency (more trustworthy than software timestamps because it includes drivers and hardware).

**7.10 GPU vs. CPU inference**
GPUs: parallel compute for training and large-batch inference. CPUs: single-stream low latency. Neural-synthesizer inference is a "small model, high frequency" workload where CPU avoids data-transfer overhead — which is why ~30 ms CPU inference is feasible and §4.6 stresses "no GPU required."

**7.11 seeds.yaml and random seeds**
Deep-learning training is full of randomness (initialization, data shuffling, dropout); fixing random seeds makes every run bit-for-bit reproducible. The paper's Appendix B specifies seed management: the repository's seeds.yaml is the central registry, covering model training, UMAP initialization, bootstrap resampling, and task-order randomization (values locked at code release).

**7.12 GitHub repository and THIRD_PARTY_NOTICES**
Standard open-source-release practice (Appendix B, §8.5): the code license (GPL-3.0) lives in its own file; third-party components' license terms (CLAP code MIT, weights CC-BY 4.0, etc.) are documented item by item in THIRD_PARTY_NOTICES to avoid the legal conflict of "main code GPL but dependencies not redistributable." Copyright-protected training audio is not distributed with the repository — only metadata plus extraction scripts. This repository (github.com/AkiroMusic/Botanica-OS) is organized accordingly.

---

## 8. Licensing, ethics, and scholarly publishing

**8.1 GPL-3.0**
Version 3 of the GNU General Public License (§8.5): strong copyleft — derivative works that use or modify the code must also be released under the GPL. Friendly to the goal of "community self-hosting and modification"; but compatibility with certain third-party weight terms must be checked item by item (which is exactly why THIRD_PARTY_NOTICES exists — the GPL-3.0 vs. CLAP-weights relationship is documented there).

**8.2 The CC license family (CC0 / CC-BY / CC-BY-NC)**
Creative Commons licenses: CC0 = all rights waived (public domain); CC-BY = any use with attribution; CC-BY-NC = adds a "non-commercial" restriction. §4.5's handling: Freesound items are recorded per item; CC-BY entries carry an attribution list; NC entries, after item-by-item review, were used only for local experiments and paper figures and do not enter the training set distributed with open weights. The boundary between model training and CC-NC ("does training count as use?") remains contested — the paper takes the conservative route.

**8.3 Informed consent and the right to withdraw**
The core of research ethics: participants voluntarily join after adequate disclosure (purpose, data use, recording, retention) and may quit and withdraw their data at any time. Three places in the paper: study participants (§6.2, written informed consent); reward raters (§4.3, an additional data-use consent); workshop observations (§3.3, all signed written consent, archived). Template: study/consent-form-template.md in the repository.

**8.4 IRB / ethics review**
The Institutional Review Board: an institutional ethics committee that approves protocols before research begins. An independent researcher without an institutional affiliation typically substitutes written informed consent, data minimization, and withdrawal mechanisms — this study took the latter route (the acknowledgments and §6.2 express everything in terms of informed consent and withdrawal rights).

**8.5 Double-blind review and the author pseudonym**
Double-blind review: reviewers and authors do not know each other's identities; submissions omit identity-revealing information. The public repository matches the paper's anonymization policy: the author is referred to as **AkiroMusic** throughout, and the legal name appears only in the paper's formal version.

**8.6 ORCID and corresponding author**
ORCID: the international unique researcher identifier (16 digits), usually required by submission systems. Corresponding author: the responsible author who communicates with the journal/reviewers (marked * in the paper's byline). This is a single-author study: AkiroMusic (akiromusic@qq.com), independent researcher.

**8.7 Independent research and funding statement**
This study is independent research receiving no external funding; the author-contribution statement covers the entire work by a single author (conceptualization, system design, implementation, training, study, analysis, and writing). For submission without funding: "This research received no specific grant from any funding agency."

**8.8 arXiv / DOI / preprints**
arXiv: the preprint platform (many citations in the paper, e.g., MusicGen and RAVE, use arXiv IDs); DOI: the permanent digital identifier of a document. All 45 references in the paper have been verified for authenticity: 16 journal/conference papers carry Crossref-verified DOIs, arXiv entries carry arXiv IDs (arXiv:XXXX.XXXXX), and the remainder are books and DOI-less conference/working papers (located by publisher and year).

---

*This glossary is maintained alongside the paper; where it and the paper disagree, the paper prevails (Table 7 as the single source of numbers).*
