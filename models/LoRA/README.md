# LoRA Detection
## LoRA Objects
We generate detectable LORA's using the `lora_conf.py` file.  I have attached a `sample_lora_conf.py` file.  This is so that we can have a template to work from.

```python
{
    f'{BASE_DIR}/models/LoRA/LORA_FOLDER':{
        "weight_name":"LORA_FILE_NAME.safetensors",
        "keywords":["KEYWWORD1","ANOTHER KEYWORD", "THIS KEYWORD TRIGGERS"],
        },
}
```

## Object Explanation
### Directory to LoRA Folder
This is the system path to the folder of a LoRA we have just added.
If I have a LoRA for Better Hand Creation, then (for organization purposes) I would put the `.safetensors` file in a `better_hand_creation` folder.

So for this example, we would put `f'{BASE_DIR}/models/LoRA/better_hand_creation'` as the first key.

### Weight Name
The Weight name is the actual name of the `.safetensors` file.  
According to our example, I'd put `better_hand_creation.safetensors`  as the `weight_name`

### Keywords
As of right now, we keep our trigger key words in an array.  Majority of LoRA's have keywords to trigger them, this is where we would put a list of all keywords that trigger the LoRA.  
For our example, if the words `"Better Hands"` would trigger our LoRA, then I would put those in the list.  
So, I would add: `["Better Hands"]`.

### Base Model
This is how we know which Model the LoRA is for.  This directly corresponds to the model's name in `settings.py`

### Scale
This is the scale for how aggressive the weight should be applied.  This is usually recommended by the LoRA provider, and it is a digit between 0 and 1.

### Is Style
This is a setting I use for other projects.  For this repository, this just enables the lora to be accessible from muliple api endpoints.

## Code
### Creating LoRA objects
In the code, we use `factories/lora_factory.py` to create LoRA objects.

To create a LoRA, it might look something like this:
```python
def create(json, path):
        return LoRA(
            path=path,
            weight_name=json.get("weight_name"),
            keywords=json.get("keywords"),
            base_model=json.get("model"),
            scale=json.get("scale"),
            is_style=json.get("is_style"),
        )
```

### Using LoRA objects
We use LoRA objects with the following example:
```python
for lora in lora_to_use:
    lora_path = lora.path[0]
    lora_weight_name = lora.weight_name
    self.pipeline.load_lora_weights(
        lora_path,
        weight_name=lora_weight_name,
        )
    self.pipeline.fuse_lora()
```

### Removing LoRA's
To remove LoRA's, we need to `unfuse` and `unload` them from our pipeline.  We do so with the following example:

```python
self.pipeline.unfuse_lora()
self.pipeline.unload_lora_weights()
```