# Perceiver
The goal for this project is to implement the Perceiver architectural family.

## What?
[] The first iteration of this family was introduced in the paper [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206). The paper illustrates that since transformers make few assumptions about the relationship between input elements, they are general enough to process inputs from many different modalities. Since self-attention must produce an (input length x input length) shaped attention pattern, this component scales quadratically with the size of the input sequence, and is therefore prohibitive for long sequence lengths from inputs such as images (processed in rastor-order) or point clouds as an example. To get around this problem, the authors of this paper propose that inputs instead be cross-attended to by a fixed-length sequence of latent vectors which start in some learned initial state, and effectively pull information from the input space into some latent-space which is processed by a series of standard transformer layers. This allows us to handle inputs with much longer sequence lengths by keeping one side of the attention matrix fixed, resulting in an attention pattern over the input which has sub-linear scaling wrt the input's length.

[] The second iteration of this family was introduced in the paper [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795), and extends the original architecture by reframing the intial version in terms of an encoder-(processor)-decoder style model. They frame the process of using cross-attention to pull information from input space into latent space as an *encoder*. This is followed by a series of transformer layers operating in latent space (*processor*), and is completed with a new, fixed-length sequence of learned vectors that exist in the output space, and cross-attend to the latent space to produce a final sequence of output vectors, *decoding* the latent results into the appropriate output space. This allows the model to take in arbitratry inputs and produce arbitrary outputs while maintaining efficient scaling properties for each.

## Why?
The primary goal for this project is to gain some familiarity with the Perceiver and use it as a building block for making visually-conditioned langauge models like [Flamingo](https://arxiv.org/abs/2204.14198) or (presumably) GPT-4.

In addition (and in general), I would like to gain familiarity with models which operate on multiple modalities and processes them in latent space. This approach seems super promising to me as a research direction, having seen models like Gato from [A Generalist Agent](https://arxiv.org/abs/2205.06175) perform well on multiple tasks with a single decoder-only Transformer model.
