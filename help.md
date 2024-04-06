- if CUDA_HOME environment variable is not set.
    
    1. `conda install -c nvidia cuda`
    2. `which nvcc`

- couldn't connect to 'https://huggingface.co'
    
    ```
    export HF_ENDPOINT=https://hf-mirror.com
    ```

- install flash-attention

    - make sure ninja is installed

        1. `ninja --version`
        2. `echo $?`
        3. if return exit code 0 then done.
        4. if not `pip uninstall -y ninja && pip install ninja`
    - install flash-attention by pip

        ```
        pip install flash-attn --no-build-isolation
        ```

        - tested verison 'flash-attn==2.5.6'

- multi-GPU inference
    
    ```
    # /models/causallm.py
    self.model = LLM(model=model_path, tensor_parallel_size=2, **self.model_kwargs)
    ```
