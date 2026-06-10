# Set Transformer with Partial Observation of Large Latent Sets

A self-contained mathematical reference. Part I states the Set Transformer
(Lee et al., 2019) precisely, with the invariance/equivariance properties that
make it correct *by construction*. Part II extends it to the regime you
described: a true set far larger than any single observation
(e.g. observing $n=8{,}000$ of $N=200{,}000$), multiple **overlapping**
partially-observed sets, and an explicit term for *how much* was observed.

Notation throughout: a set/multiset of feature vectors is a matrix
$X \in \mathbb{R}^{n \times d}$ whose rows $x_i \in \mathbb{R}^d$ are the
elements; $n$ is the number of elements, $d$ the feature dimension. $P \in \{0,1\}^{n\times n}$ denotes a permutation matrix.

---

## Part I — The Set Transformer

### 1. Invariance and equivariance

A function $f$ on sets is **permutation invariant** if reordering the elements
does not change the output:

$$
f(PX) = f(X)\quad\text{for every permutation } P .
$$

A function is **permutation equivariant** if reordering the inputs reorders the
outputs identically:

$$
f(PX) = P\, f(X)\quad\text{for every permutation } P .
$$

The Set Transformer is built so that its *feature extractor* is equivariant and
its *pooling* is invariant; an equivariant map followed by an invariant pool is
invariant. This is the structural guarantee — it holds for **any** weights, so
invariance never has to be learned from order augmentation.

### 2. Attention

Scaled dot-product attention with queries $Q\in\mathbb R^{n_q\times d}$, keys
$K\in\mathbb R^{n_v\times d}$, values $V\in\mathbb R^{n_v\times d}$:

$$
\operatorname{Att}(Q,K,V) = \operatorname{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d}}\Big)\,V .
$$

For a single query $q$ the output is a convex combination of the values,

$$
\operatorname{Att}(q,K,V)=\sum_{i} a_i v_i,\qquad
a_i=\frac{\exp(q^\top k_i/\sqrt d)}{\sum_j \exp(q^\top k_j/\sqrt d)} .
$$

**Key fact.** $\sum_i a_i v_i$ is invariant to jointly permuting the
$(k_i,v_i)$ pairs, because addition is commutative. This single fact is the
source of every invariance below.

Multihead attention runs $h$ independent projections and concatenates:

$$
\operatorname{MH}(Q,K,V)=\operatorname{concat}(O_1,\dots,O_h)\,W^O,\quad
O_j=\operatorname{Att}(QW_j^Q,\,KW_j^K,\,VW_j^V).
$$

### 3. The four blocks

**Multihead Attention Block** (transformer block; $X$ attends to $Y$, with
row-wise feed-forward $\mathrm{rFF}$ and LayerNorm):

$$
\operatorname{MAB}(X,Y)=\operatorname{LN}\!\big(H+\mathrm{rFF}(H)\big),\qquad
H=\operatorname{LN}\!\big(X+\operatorname{MH}(X,Y,Y)\big).
$$

$\operatorname{MAB}(X,Y)$ is **equivariant in $X$** and **invariant to permutations of $Y$** (because $Y$ sits in the key/value position).

**Set Attention Block** — self-attention among elements, modelling pairwise
interactions; equivariant; cost $O(n^2)$:

$$
\operatorname{SAB}(X)=\operatorname{MAB}(X,X).
$$

**Induced Set Attention Block** — introduces $m$ learnable *inducing points*
$I\in\mathbb R^{m\times d}$ ($m\ll n$) as a bottleneck; equivariant; cost
$O(nm)$:

$$
\operatorname{ISAB}_m(X)=\operatorname{MAB}(X,\,H)\in\mathbb R^{n\times d},
\qquad H=\operatorname{MAB}(I,\,X)\in\mathbb R^{m\times d}.
$$

The inducing points first attend to the set ($H$ compresses $n\to m$), then the
set attends back to $H$. This is a low-rank surrogate for full self-attention.

**Pooling by Multihead Attention** — the *learnable, invariant* aggregation
that replaces sum/mean/max. With $k$ learnable *seed* queries
$S\in\mathbb R^{k\times d}$ and encoder features $Z\in\mathbb R^{n\times d}$:

$$
\operatorname{PMA}_k(Z)=\operatorname{MAB}\!\big(S,\ \mathrm{rFF}(Z)\big)\in\mathbb R^{k\times d}.
$$

Because $Z$ is in the key/value position, $\operatorname{PMA}_k$ is
**permutation invariant** in the set. With $k=1$ it returns a single pooled
vector; $k>1$ yields several outputs (e.g. for amortized clustering). The
weights are content-dependent — a learned, query-conditioned weighted average,
generalizing the fixed mean of Deep Sets.

### 4. Full model and its invariance

$$
\operatorname{Encoder}(X)=\big(\operatorname{ISAB}_m\big)^{\circ L}(X)\quad\text{(equivariant)},
$$
$$
\operatorname{Decoder}(Z)=\rho\Big(\mathrm{rFF}\big(\operatorname{SAB}(\operatorname{PMA}_k(Z))\big)\Big)\quad\text{(invariant)},
$$
$$
f(X)=\operatorname{Decoder}\big(\operatorname{Encoder}(X)\big).
$$

The encoder is a composition of equivariant blocks, hence equivariant; the
decoder begins with the invariant $\operatorname{PMA}_k$, hence the whole $f$ is
permutation invariant. The optional $\operatorname{SAB}$ after pooling models
interactions among the $k$ pooled outputs (trivial when $k=1$).

---

## Part II — Partial observation of a much larger latent set

### 5. Sampling model

Let the **true latent set** be $\mathcal S=\{x_1,\dots,x_N\}$ with $N=|\mathcal S|$
large, and the **observed sample** $\mathcal O\subseteq\mathcal S$ with
$n=|\mathcal O|\ll N$ (your example: $n=8{,}000$, $N=200{,}000$, coverage
$c=n/N=0.04$).

Define inclusion indicators $Z_i=\mathbb 1[x_i\in\mathcal O]$ with **first-order
inclusion probabilities** $\pi_i=\mathbb P(Z_i=1)=\mathbb E[Z_i]$ and
second-order $\pi_{ij}=\mathbb P(Z_i=1,Z_j=1)$. For uniform simple random
sampling without replacement of size $n$,

$$
\pi_i=\frac{n}{N},\qquad \pi_{ij}=\frac{n(n-1)}{N(N-1)} .
$$

The general (informative / non-uniform) case keeps $\pi_i$ arbitrary but known
up to a constant; the uniform case is the special case $\pi_i\equiv n/N$.

### 6. Pooling as estimation of a population statistic

The quantity you would compute if you saw all of $\mathcal S$ is a population
total or mean of the per-element embedding $\phi$,

$$
T=\sum_{i\in\mathcal S}\phi(x_i),\qquad
\bar\phi=\frac1N\sum_{i\in\mathcal S}\phi(x_i).
$$

The naive observed sum $\sum_{i\in\mathcal O}\phi(x_i)$ is biased downward by the
coverage. The **Horvitz–Thompson** estimator removes the bias by inverse-
probability weighting:

$$
\hat T_{\mathrm{HT}}=\sum_{i\in\mathcal O}\frac{\phi(x_i)}{\pi_i}
=\sum_{i\in\mathcal S}\frac{Z_i}{\pi_i}\phi(x_i),
\qquad
\mathbb E\big[\hat T_{\mathrm{HT}}\big]
=\sum_{i\in\mathcal S}\frac{\pi_i}{\pi_i}\phi(x_i)=T .
$$

Under uniform sampling this is the explicit **number-of-observations
correction**:

$$
\hat T_{\mathrm{HT}}=\frac{N}{n}\sum_{i\in\mathcal O}\phi(x_i).
$$

The **Hájek** (ratio) estimator targets the *mean* and is self-normalizing
(robust to unknown sampling scale):

$$
\hat{\bar\phi}_{\mathrm{Haj}}
=\frac{\sum_{i\in\mathcal O}\phi(x_i)/\pi_i}{\sum_{i\in\mathcal O}1/\pi_i}
\;\xrightarrow{\ \pi_i\equiv n/N\ }\;
\frac1n\sum_{i\in\mathcal O}\phi(x_i).
$$

So **mean-pooling is already a design-consistent estimator of the population
mean** and is approximately invariant to $n$, whereas **sum-pooling estimates a
total** and scales with $n$. Choose the estimator by whether your target scales
with $N$.

### 7. Sampling-corrected attention pooling

$\operatorname{PMA}$ produces a self-normalizing weighted average over the
*observed* set,

$$
\text{pool}=\sum_{i\in\mathcal O}a_i v_i,\qquad
a_i=\frac{\exp(s^\top k_i/\sqrt d)}{\sum_{j\in\mathcal O}\exp(s^\top k_j/\sqrt d)} .
$$

To make this estimate the **population** attention-weighted mean rather than the
sample one, add $-\log\pi_i$ to each logit (a Hájek correction inside the
softmax):

$$
\tilde a_i=
\frac{\exp\!\big(s^\top k_i/\sqrt d-\log\pi_i\big)}
     {\sum_{j\in\mathcal O}\exp\!\big(s^\top k_j/\sqrt d-\log\pi_j\big)}
=\frac{(1/\pi_i)\,\exp(s^\top k_i/\sqrt d)}
       {\sum_{j\in\mathcal O}(1/\pi_j)\,\exp(s^\top k_j/\sqrt d)} .
$$

Under uniform $\pi_i$ the term $-\log\pi_i$ is constant and cancels in the
softmax, recovering ordinary $\operatorname{PMA}$ — consistent with mean-pooling
already being design-consistent. Write $\operatorname{PMA}^{\pi}$ for this
inclusion-reweighted pooling. (For a population *total* via attention, use the
HT form $\sum_{i\in\mathcal O} a_i v_i/\pi_i$.)

### 8. The cardinality term: conditioning on $(n,N)$

Self-normalizing pooling deliberately discards $n$. If the target depends on how
much was seen — on $n$, $N$, or coverage $c=n/N$ — reinject it. With pooled
vector $z=\operatorname{PMA}_1^{\pi}(\cdot)\in\mathbb R^d$, form a **cardinality
embedding** from whichever size descriptors are identifiable,

$$
u=\big(n,\;N,\;n/N,\;\log n,\;\log N,\;\log(N/n)\big),\qquad
e=\operatorname{MLP}_{\mathrm{card}}(u)\in\mathbb R^{d},
$$

and combine with the set summary by **FiLM** modulation (usually best),
concatenation, or addition:

$$
(\gamma,\beta)=\operatorname{MLP}_{\mathrm{card}}(u),\qquad
z'=\gamma\odot z+\beta,
\qquad\text{then}\qquad
\hat y=\rho(z').
$$

FiLM lets the network rescale/shift the set summary according to coverage.

**Identifiability of $N$.** HT and the $N/n$ factor require $N$. If $N$ is
unknown: (i) supply $N$ or $c$ as a covariate when available; (ii) estimate it
by capture–recapture across overlapping observations (§11); or (iii) fall back
to mean / Hájek pooling, which need only relative $\pi_i$. State which regime
applies.

### 9. Sampling uncertainty (finite-population correction)

Because $z$ is built from a sample, it is an estimate whose variance shrinks as
$n\to N$. For a mean-type estimator under SRS without replacement,

$$
\operatorname{Var}\!\Big(\tfrac1n\!\sum_{i\in\mathcal O}\phi_i\Big)
=\underbrace{\Big(1-\tfrac{n}{N}\Big)}_{\text{fpc}}\frac{S^2}{n},
\qquad S^2=\frac{1}{N-1}\sum_{i\in\mathcal S}\big(\phi_i-\bar\phi\big)^2 .
$$

The **finite-population correction** $(1-n/N)\to0$ as $n\to N$ (no uncertainty
once the whole set is observed) and $\to1$ when $n\ll N$. A predictive head can
carry this structure explicitly:

$$
\hat y\sim\mathcal N\!\Big(\mu_\theta(z'),\;
\Sigma_\theta(z')\cdot\tfrac{1-n/N}{n}\Big),
$$

so the model is more confident the more of the population it has seen. This ties
the number of observations directly to calibrated uncertainty.

---

## Part III — Multiple overlapping sets

### 10. Membership structure

Let $\mathcal U$ be the universe of distinct elements, $|\mathcal U|=M$. There
are $S$ latent sets $\mathcal S_1,\dots,\mathcal S_S\subseteq\mathcal U$ with
$|\mathcal S_s|=N_s$, encoded by an **incidence matrix**
$B\in\{0,1\}^{S\times M}$, $B_{si}=\mathbb 1[x_i\in\mathcal S_s]$. Overlap is
measured by co-membership $\langle B_s,B_t\rangle=\sum_i B_{si}B_{ti}$ or Jaccard

$$
J_{st}=\frac{|\mathcal S_s\cap\mathcal S_t|}{|\mathcal S_s\cup\mathcal S_t|} .
$$

Each set is observed only partially: $\mathcal O_s\subseteq\mathcal S_s$,
$|\mathcal O_s|=n_s$, with $n_s/N_s$ small (your ≈5%), and per-set inclusion
probabilities $\pi_{s,i}$.

**Shared elements.** The same element $x_i$ may be observed in several sets. Use
one **tied** element encoder $\phi$ so $x_i$ contributes a consistent embedding
everywhere, plus an optional identity embedding $\eta_i$ so co-occurrence across
sets is recognizable:

$$
h_{s,i}=\phi(x_i)+\eta_i .
$$

### 11. Two-level (hierarchical) attention

**Level 1 — within set** (element $\to$ set summary), for each $s$:

$$
H_s=\operatorname{Encoder}\big(\{h_{s,i}:i\in\mathcal O_s\}\big),\qquad
g_s=\rho_{\mathrm{set}}\Big(\operatorname{PMA}_1^{\pi}(H_s),\;e(n_s,N_s)\Big)\in\mathbb R^d .
$$

Each set summary $g_s$ is thus sampling-corrected (§7) and coverage-aware (§8).

**Level 2 — across sets** (set $\leftrightarrow$ set interaction). Stack the
summaries $G=[g_1;\dots;g_S]\in\mathbb R^{S\times d}$ and model interactions with
self-attention,

$$
G'=\operatorname{SAB}(G+\Psi),
$$

where $\Psi=[\psi_1;\dots;\psi_S]$ are optional **set-identity encodings** (omit
them if the sets are exchangeable). The **overlap structure** enters as an
attention bias so that more-overlapping sets attend more strongly:

$$
\text{logit}_{st}\;\mathrel{+}=\;b(J_{st})
\quad\text{(a graph-attention prior on the set-overlap graph).}
$$

**Readout.** Global output $\hat y=\rho_{\mathrm{glob}}(\operatorname{PMA}_1(G'))$,
or per-set outputs $\hat y_s=\rho(g'_s)$, each with the optional fpc-scaled
predictive variance $\propto (1-n_s/N_s)/n_s$.

### 12. Element-level cross-set coupling (complementary)

Interactions also flow through *shared elements*. Model the bipartite graph
$\mathcal U\leftrightarrow\{\mathcal S_s\}$ and alternate messages — this is the
$\operatorname{ISAB}$ idea with the **sets themselves as inducing points**
(semantically meaningful rather than learned):

$$
\text{element}\to\text{set:}\quad
g_s=\operatorname{MAB}\big(q_s,\ \{h_{s,i}:i\in\mathcal O_s\}\big),
$$
$$
\text{set}\to\text{element:}\quad
h_{s,i}\leftarrow\operatorname{MAB}\big(h_{s,i},\ \{g_t : x_i\in\mathcal S_t\}\big).
$$

Alternate for $L$ rounds. Information propagates between sets through the
elements they share, capturing inter-set interaction at the finest granularity.
Cost is $O\big(\sum_s n_s\,d\big)$ per round (linear in total observations),
since each element attends only to the sets it belongs to.

### 13. Estimating unknown $N_s$ from overlap (capture–recapture)

Overlapping observations let you estimate set sizes when they are unknown. For
two observations $\mathcal O_s,\mathcal O_t$ of the same set (or two passes), the
Lincoln–Petersen estimator is

$$
\hat N\approx\frac{|\mathcal O_s|\,|\mathcal O_t|}{|\mathcal O_s\cap\mathcal O_t|},
$$

supplying the $N_s$ needed by the HT correction (§6) and the cardinality term
(§8). Standard assumptions apply: closed population, equal catchability (or
explicitly modelled heterogeneity), independent observations.

### 14. Symmetry group — what "invariant by construction" means here

The symmetry is no longer a single $S_n$. With $S$ observed sets of sizes
$n_1,\dots,n_S$:

- Invariance to permuting elements **within** each observed set is always
  required: the group $\prod_{s=1}^S S_{n_s}$.
- If the sets are themselves **exchangeable** (no identities), add invariance to
  permuting the sets, giving the wreath product $S_n\wr S_S$ in the equal-size
  case. With set-identity encodings $\psi_s$, you keep only the within-set
  product $\prod_s S_{n_s}$ (equivariant, not invariant, across sets).

The architecture enforces the within-set part by construction (equivariant
encoders + invariant/attention pooling), and the across-set part by construction
iff set-identity encodings are omitted (or the cross-set block is kept
equivariant and closed with an invariant pool).

---

## Part IV — Reference forward pass

Inputs: observed sets $\{\mathcal O_s\}_{s=1}^S$, inclusion probabilities
$\{\pi_{s,i}\}$, sizes $\{n_s\}$, known-or-estimated $\{N_s\}$.

1. **Element embeddings**  $h_{s,i}=\phi(x_i)+\eta_i$.
2. **Within-set encoder**  $H_s=(\operatorname{ISAB}_m)^{\circ L}\big(\{h_{s,i}\}_{i\in\mathcal O_s}\big)$  — equivariant, $O(n_s m)$.
3. **Sampling-corrected, coverage-aware pooling**  $g_s=\rho_{\mathrm{set}}\big(\operatorname{PMA}_1^{\pi}(H_s),\,e(n_s,N_s)\big)$.
4. **Cross-set interaction**  $G'=\operatorname{SAB}(G+\Psi)$ with overlap bias $b(J_{st})$ (and/or the element-level coupling of §12).
5. **Readout**  $\hat y=\rho_{\mathrm{glob}}(\operatorname{PMA}_1(G'))$ or per-set $\hat y_s=\rho(g'_s)$, with optional predictive variance $\propto(1-n_s/N_s)/n_s$.

**Guarantees of this pipeline.** It is invariant to within-set element order
*by construction*; it is sampling-consistent (HT / Hájek) for the population
statistics; it is explicitly conditioned on how much of each set was observed
through $e(n_s,N_s)$; and it models inter-set interaction both at the set level
(self-attention with an overlap prior) and through shared elements.

---

## Design cheat-sheet

| Goal | Mechanism | Key equation |
|---|---|---|
| Order invariance | symmetric / attention pooling | $\sum_i a_i v_i$ unchanged under reordering |
| Element interactions | SAB / ISAB encoder | $\operatorname{SAB}(X)=\operatorname{MAB}(X,X)$ |
| Scale to large $n$ | $m$ inducing points | $\operatorname{ISAB}_m$, cost $O(nm)$ |
| Learned pooling | seed-query attention | $\operatorname{PMA}_k(Z)=\operatorname{MAB}(S,\mathrm{rFF}(Z))$ |
| Estimate population total | HT reweighting | $\hat T_{\mathrm{HT}}=\sum_{i\in\mathcal O}\phi_i/\pi_i$ |
| Estimate population mean | mean / Hájek pooling | $\frac1n\sum_{i\in\mathcal O}\phi_i$ |
| Correct attention for sampling | logit bias | $\text{logit}_i\mathrel{+}=-\log\pi_i$ |
| Condition on coverage | cardinality FiLM | $z'=\gamma(u)\odot z+\beta(u)$, $u\ni n,N,n/N$ |
| Sampling uncertainty | finite-pop. correction | $\operatorname{Var}=(1-n/N)\,S^2/n$ |
| Overlapping sets | two-level attention | within-set $\operatorname{PMA}^\pi$ + cross-set $\operatorname{SAB}$ |
| Inter-set coupling | shared-element messages | element$\leftrightarrow$set $\operatorname{MAB}$ |
| Unknown $N_s$ | capture–recapture | $\hat N\approx n_s n_t/|\mathcal O_s\cap\mathcal O_t|$ |

## Reference

Lee, Lee, Kim, Kosiorek, Choi, Teh. *Set Transformer: A Framework for
Attention-based Permutation-Invariant Neural Networks.* ICML 2019.
(Deep Sets baseline: Zaheer et al., NeurIPS 2017.) §§5–14 above are a modelling
extension, not part of the original papers.