code failed beacuse of memory util was very much


before runing the code in expensive compute make sure all code works and they are reliable

The run is invalid because of a training-set bug in my notebook, not because of the hardware.



A. Label-free monitoring of latent reasoning.
- Fit an anomaly detector (for example a Mahalanobis or density model) on benign latents only, with no misbehaviour labels, and test whether it flags hint use.
- Why it matters: real deployments won't have labelled misbehaviour.
- The only related paper I found, Ulterior Motives, is supervised.

B. Cross-format monitor transfer.
- Train a monitor on the explicit-CoT model, where labels are easy to get and the trace is readable. Then apply it to the COCONUT model fine-tuned from the same base.
- If it transfers, you can build monitors for latent models without needing latent-labelled data.


hident sidetask capacity vs hte number of latents 