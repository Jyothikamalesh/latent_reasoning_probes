"""Validation of analysis/latent_patching.py on a tiny random GPT-2 (protocol Amendment 1, item 2). Run: python -m pytest prosqa_monitor/tests/test_latent_patching.py"""
import os, sys, unittest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))
import latent_patching as lp

PAD, BOT, EOT, LAT, N = 0, 1, 2, 3, 4


def tiny():
    torch.manual_seed(0)
    m = GPT2Config(vocab_size=40, n_positions=64, n_embd=32, n_layer=2, n_head=2, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1)
    return GPT2LMHeadModel(m).eval()


PROMPTS = [[5, 6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17]]       # different lengths -> left padding
HINTS = [3, 2]                                                      # trailing hint-line tokens
TAILS = [[EOT, 20, 21, 22], [EOT, 23, 24]]


def batch(prompts=PROMPTS, hints=HINTS, tails=TAILS):
    return lp.make_batch(prompts, hints, tails, N, PAD, BOT, LAT)


class T(unittest.TestCase):
    def setUp(self): self.m = tiny()

    def test_identity_patch(self):
        bt = batch(); base = lp.run(self.m, bt)
        pat = lp.run(self.m, bt, patch={i: base['thoughts'][:, i] for i in range(N)})
        self.assertTrue(torch.allclose(base['logp'], pat['logp'], atol=1e-6))
        self.assertTrue(torch.allclose(base['thoughts'], pat['thoughts'], atol=1e-6))

    def test_unhinted_to_unhinted_changes_nothing(self):
        un = [p[:len(p) - h] for p, h in zip(PROMPTS, HINTS)]
        bt = batch(un, [0, 0]); a = lp.run(self.m, bt)
        b = lp.run(self.m, bt, patch={2: a['thoughts'][:, 2]})
        self.assertTrue(torch.allclose(a['logp'], b['logp'], atol=1e-6))

    def test_only_intended_latent_replaced(self):
        bt = batch(); base = lp.run(self.m, bt); donor = torch.randn(2, 32)
        out = lp.run(self.m, bt, patch={2: donor})
        self.assertTrue(torch.allclose(out['thoughts'][:, 2], donor))                     # the patched slot's input is the donor
        self.assertTrue(torch.allclose(out['thoughts'][:, :2], base['thoughts'][:, :2]))   # earlier slots untouched
        self.assertFalse(torch.allclose(out['thoughts'][:, 3], base['thoughts'][:, 3]))    # later slot recomputed

    def test_downstream_recomputed(self):
        bt = batch(); base = lp.run(self.m, bt); out = lp.run(self.m, bt, patch={0: torch.randn(2, 32)})
        for i in range(1, N): self.assertFalse(torch.allclose(out['thoughts'][:, i], base['thoughts'][:, i]))
        self.assertFalse(torch.allclose(out['logp'], base['logp']))

    def test_eval_no_grad_deterministic(self):
        bt = batch(); a, b = lp.run(self.m, bt), lp.run(self.m, bt)
        self.assertTrue(torch.equal(a['logp'], b['logp']))
        self.m.train()
        with self.assertRaises(AssertionError): lp.run(self.m, bt)
        self.assertFalse(a['logp'].requires_grad)

    def test_hint_span_masking(self):
        bt = batch(); a = bt['a']
        for b, h in enumerate(HINTS):
            idx = bt['block'][b].nonzero().flatten().tolist()
            self.assertEqual(idx, list(range(a - 1 - h, a)))                              # hint tokens + <bot>, nothing else
            self.assertEqual(bt['ids'][b, a - 1].item(), BOT)
            self.assertEqual(bt['ids'][b, a - 1 - h:a - 1].tolist(), PROMPTS[b][-h:])
        base = lp.run(self.m, bt)
        self.assertFalse(torch.allclose(base['logp'], lp.run(self.m, bt, block_answer=True)['logp']))
        self.assertFalse(torch.allclose(base['logp'], lp.run(self.m, bt, block_latents_from=0)['logp']))
        # blocking from a slot after the last one changes nothing
        self.assertTrue(torch.allclose(base['logp'], lp.run(self.m, bt, block_latents_from=N)['logp']))

    def test_hint_span_len(self):
        enc = lambda s: [ord(c) % 30 + 5 for c in s]
        self.assertEqual(lp.hint_span_len(enc, 'abc\nHint: x', 'abc\n'), 7)
        with self.assertRaises(AssertionError): lp.hint_span_len(lambda s: [len(s)] + enc(s)[1:], 'abc\nH', 'abc\n')

    def test_multitoken_scoring_matches_direct_forward(self):
        bt = batch(); out = lp.run(self.m, bt); a = bt['a']
        emb = self.m.get_input_embeddings()(bt['ids'])
        full = torch.cat([emb[:, :a], out['thoughts'], emb[:, a + N:]], 1)                # latent slot inputs = the recorded thoughts
        att, pos = bt['att'], bt['pos']
        logits = self.m(inputs_embeds=full, attention_mask=att, position_ids=pos).logits[:, a + N:]
        ref = torch.log_softmax(logits[:, :-1], -1).gather(-1, bt['tail_ids'][:, 1:, None])[..., 0]
        self.assertTrue(torch.allclose((ref * out['mask']).sum(1), lp.sentence_logprob(out), atol=1e-4))

    def test_padding_invariance(self):
        both = lp.sentence_logprob(lp.run(self.m, batch()))
        solo = lp.sentence_logprob(lp.run(self.m, batch(PROMPTS[1:], HINTS[1:], TAILS[1:])))
        self.assertTrue(torch.allclose(both[1:], solo, atol=1e-4))

    def test_margin_sign_and_consistency(self):
        tt, tw = [[EOT, 20, 21], [EOT, 20, 21]], [[EOT, 25, 26, 27], [EOT, 25, 26, 27]]
        M, _ = lp.margin(self.m, PROMPTS, HINTS, tt, tw, N, PAD, BOT, LAT)
        s_t = lp.sentence_logprob(lp.run(self.m, batch(tails=tt))); s_w = lp.sentence_logprob(lp.run(self.m, batch(tails=tw)))
        self.assertTrue(torch.allclose(M, s_t - s_w))


if __name__ == '__main__': unittest.main()
