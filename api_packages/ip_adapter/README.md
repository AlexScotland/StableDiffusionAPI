# IP Adapters
This package implements IP Adapter API endpoints.

This is utilizing [Tencent AI Labs IP Adapter](https://github.com/tencent-ailab/IP-Adapter)

# Installation
to use this package, the following is necessary:
 - [ ] Install the Requirements
 - [ ] Install the Models

# Requirements Installation
1. pip install git+https://github.com/tencent-ailab/IP-Adapter.git
or
1. pip install -r .requirements.txt

# Model Installation
1. `cd api_packes/ip_adapter`
2. `git lfs install`
3. `git clone https://huggingface.co/h94/IP-Adapter`
4. `mv IP-Adapter/models api_packes/ip_adapter/models`
5. `mv IP-Adapter/sdxl_models api_packes/ip_adapter/sdxl_models`