Ah, understood — you're asking about mechanistic interpretability tools for modern reasoning models, not specifically COCONUT models.

The distinction matters:
- CoT (Chain-of-Thought): The model generates reasoning as ordinary text tokens. The reasoning is visible in its output sequence, even if it isn't necessarily a faithful account of its internal computation.
- COCONUT (Continuous Thought): The model carries reasoning forward through continuous hidden-state representations rather than generating ordinary natural-language reasoning tokens at each step. Those latent steps aren't directly readable as text.
- Reasoning models: Models such as o-series or other models trained for extended reasoning may generate visible or hidden reasoning, depending on the model and interface. They are not automatically COCONUT models. A model can reason extensively in token space while still using standard transformer activations internally.
So, latent reasoning is broader than COCONUT. Even a conventional CoT reasoning model has internal representations that aren't visible in its written chain of thought.

For this, the main tools are still TransformerLens and NNsight — but there are also newer research approaches specifically aimed at extracting information from hidden reasoning activations.
Tool / approach	What it gives you
TransformerLens	Inspect residual streams, attention heads, MLPs, cache activations and perform patching.  transformerlensorg.github.io+1
NNsight	Trace and intervene on activations in Hugging Face/PyTorch reasoning models without rewriting their architecture.  nnsight.net
SAELens	Train or analyse sparse autoencoders to identify interpretable activation features.
J-Lens / R-Lens / J++ Lens	Research approaches for decoding information represented in activations but not necessarily verbalised.  GitHub

Particularly relevant research
Two examples show that this is an active research direction:
- How Does Chain of Thought Think? (AAAI 2026) uses sparse autoencoders and activation patching to investigate features involved in CoT reasoning. 
  
  ojs.aaai.org
- Reasoning Models Know What's Important, and Encode It in Their Activations investigates whether activations encode which reasoning steps matter before subsequent steps are generated. 
  
  Project Page
What I'd recommend
For a conventional reasoning model, I'd start with NNsight, not build a COCONUT-specific setup.

You can investigate:
1. Whether the answer is already represented before the model starts generating its reasoning.
2. Which internal activations encode intermediate reasoning steps.
3. Whether those representations predict errors before they appear in the written CoT.
4. Whether intervening on those activations changes the final answer.
One important distinction: a probe detecting information in an activation doesn't establish that the model uses that information causally. You need interventions to test that.

The key question: Are you interested in interpreting an open-weight reasoning model such as DeepSeek-R1 or Qwen3, where you can access its weights and hidden states, or a proprietary reasoning model such as OpenAI's o-series? That changes what we can inspect.



I think that's a worthwhile research direction — and I'd separate it from your existing ProsQA experiment.

The problem you're describing is:

> Can we build tools that make the internal representations of reasoning models interpretable, especially the information encoded in hidden states that never appears in their written chain of thought?

That's a more fundamental problem than simply detecting whether a model followed a misleading hint.

I'd call this latent-representation interpretability for reasoning models.

There is an important distinction, though: we already have tools to extract, probe, visualise and intervene on activations. What is missing is a reliable, general-purpose way to turn arbitrary internal representations into understandable concepts or reasoning structures, while validating that those explanations reflect the model's actual computation.

## Important discovery: tools closer to what you want already exist

I checked the current research landscape, and I need to correct one part of my earlier answer: there are now methods specifically designed to decode unspoken intermediate variables from model activations.

[A global workspace in language models \ Anthropic](https://images.openai.com/static-rsc-4/SBVjD7hFVTNZSMVnlNqFkDrwNKqb4tYy3uFM26Pce1rdDlRFgkdEPa19phlxXoGvYvh8rR3LimugGynkVup-UT6k-bYZPn7tbkM9N1v1fQWAiZxqVUgI2y_Oyboq5_qYk2CK0vVSt_xV0rwhoc0QQk_7fSlvvR0TVYziq-DX2TU?purpose=inline)

1\. J-Lens — Jacobian Lens

Reads intermediate activations as concepts the model is disposed to verbalise, even when it hasn't actually generated those concepts.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.transformer-circuits.pub\&sz=32)

transformer-circuits.pub



Research: Verbalizable Representations Form a Global Workspace in Language Models (2026)

[Anthropic's Global Workspace in Claude: Conscious or Not? | ThePlanetTools.ai](https://images.openai.com/static-rsc-4/Y0KFm3nCpU-UUpWptdpKSMEg7iH4eWdo3Gs1s3OOcX-De8l-xKGD5GO0Ka1OSYq3Hq1NR6eudTrmypWv2iAlDdBn6hDUlfh1kpI92sF3hELjoh4hQyXYrlcX3MpQ6-UWwkmuVQBIMsHgIkiJg6TyOadyCBvozCu3eK3WuM5KZlQ?purpose=inline)

2\. R-Lens

A refinement of J-Lens intended to improve early-layer readouts and reduce noise.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.alignmentforum.org\&sz=32)

alignmentforum.org



[Verbalizable Representations Form a Global Workspace in Language Models](https://images.openai.com/static-rsc-4/ViZ6F8SI0L0tP0gNvr37kvDyJsv0syCYoM5-HoiMl4GoqlkXhqOG-xYNT2eZ99eN3iVVSkyOU1b4m2h3zegMxysWMqvjVfe42-CRMLBIBI7y8fW5ZXjG3bQzc_VcLSPLq7F3mdFKghJtflEbHIKkaT4k_dKRXhXho0ZCok071aI?purpose=inline)

3\. J++ Lens

Improves on J-Lens using multiple context-dependent Jacobian experts and filtering. Its authors report 55.2% recall\@10 on five intermediate-variable readout tasks for Qwen3.6-27B.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



Code: [J++ Lens repository](https://github.com/safety-research/jpp_lens?utm_source=chatgpt.com)

There is also work using SAEs to identify reasoning-related features in DeepSeek-R1, and step-level SAEs intended to extract reasoning transitions rather than individual token features.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.goodfire.com\&sz=32)

Goodfire+2



So the opportunity is not that nobody has built a latent decoder. The opportunity is that these decoders are still limited, model-dependent, and not yet reliable general-purpose interpreters.

## Where I think the actual research opportunity lies

I would distinguish three levels of interpretability:

| Level                    | Question                                                              | Existing methods                                    |
| ------------------------ | --------------------------------------------------------------------- | --------------------------------------------------- |
| Decodability             | What information is present in the activation?                        | Probes, logit lens, J-Lens                          |
| Representation           | What reusable features or variables are encoded?                      | SAEs, J-Lens, R-Lens                                |
| Mechanistic faithfulness | Does the decoded information actually participate in the computation? | Activation patching, ablation, causal interventions |

The difficult part is moving from the first two to the third.

For example, imagine a reasoning model solving:

`If Alice is taller than Bob, and Bob is taller than Carol, who is shortest?`

At an intermediate point, the model might internally represent:

- Alice > Bob
- Bob > Carol
- Therefore Carol is shortest

Can we recover those intermediate variables from its activations before they appear in its written reasoning?

And can we intervene on the representation of `Bob > Carol` and reliably change the subsequent computation?

That is a much more interesting target than simply decoding the next likely word.

## A project I would actually pursue

Build a systematic evaluation framework for recovering and validating hidden reasoning states.

Not another activation viewer. Not just another SAE. A framework that answers:

> Given a model's internal activations, how accurately can we recover what it is computing, at what point can we recover it, and does that recovered information have causal relevance?

I'd structure it in four stages.

1. Establish baselines. Compare linear probes, SAEs, J-Lens, R-Lens and J++ Lens where supported.
2. Create controlled reasoning tasks. Use tasks with known intermediate variables: mathematical operations, relational reasoning, multi-hop logic and planning.
3. Recover hidden states. Attempt to decode intermediate variables before they appear in the generated CoT.
4. Validate causally. Patch, suppress or alter the recovered representation and measure whether the model's subsequent reasoning changes.

### Which models?

| Model                       | Why use it                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------ |
| Qwen3-8B                    | Practical open-weight starting point for activation extraction and experiments.      |
| DeepSeek-R1-Distill-Qwen-7B | Lets you compare a reasoning-distilled model against a more conventional Qwen model. |

I'd start with these smaller models rather than full DeepSeek-R1. The full model is substantially more expensive to instrument.

### What could make this a real contribution?

I see three possible contributions:

- A benchmark: How well can different interpretability methods recover known but unspoken intermediate reasoning variables?
- A method: A better decoder that recovers structured intermediate variables, including concepts that aren't easily represented by individual vocabulary tokens.
- A validation framework: Distinguish information that is merely decodable from information that causally contributes to subsequent reasoning.

The third is particularly important. A recent ICML 2026 study raises concerns that SAE-based reasoning features can capture superficial textual cues rather than genuine reasoning mechanisms.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://proceedings.mlr.press\&sz=32)

proceedings.mlr.press



## My recommendation

Don't immediately start building a new interpretability package.

First, reproduce J-Lens or J++ Lens on a supported model and establish exactly where existing approaches fail. Then determine whether the failure is:

- The information isn't recoverable.
- The information is recoverable but distributed across multiple features.
- The information is recoverable but not in a vocabulary-like form.
- The information is decoded correctly but the explanation isn't causally faithful.

These are four different research problems.

One possible research thesis:

Can we develop a model-agnostic framework for recovering and causally validating intermediate computational states in reasoning language models?

I think this is a more promising direction to investigate than extending the current ProsQA monitor. But the literature is already moving quickly, so the novelty would need to come from a demonstrable capability that existing lenses, probes and SAEs cannot provide.

I'd begin with a replication and failure analysis of J++ Lens on Qwen, then use those failures to define the research problem.