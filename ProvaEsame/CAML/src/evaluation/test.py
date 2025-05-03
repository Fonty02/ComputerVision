import sys
import torch
import torch.nn as nn # Importa torch.nn

from pyprojroot import here as project_root
print(project_root())
sys.path.insert(0, str(project_root()))

from src.evaluation.utils import get_test_path, get_model
from src.evaluation.eval import meta_test

from src.train_utils.trainer import train_parser
from src.models.feature_extractors.pretrained_fe import get_fe_metadata

# Importa il modello base invece del classificatore
from transformers import AutoImageProcessor
from transformers.models.siglip import SiglipVisionModel # Importa SiglipVisionModel

data_path = '../caml_universal_eval_datasets'

"""
To evaluate your model on a downstream dataset in the universal setting simply run a command like:

python src/evaluation/test.py --model CAML --gpu 4 --eval_dataset pascal_paintings  --fe_type timm:vit_base_patch16_clip_224.openai:768

We use either 
--fe_type timm:vit_base_patch16_clip_224.openai:768
--fe_type timm:vit_huge_patch14_clip_224.laion2b:1280
--fe_type timm:resnet34:512
for CLIP, Laion-2b and ResNet-34, respectively.


For MetaOptNet set

export CUDA_VISIBLE_DEVICES=7 [or alternatively another GPU id]

b/c the SVM in their code uses .cuda() rather than .to(device).

A list of datasets: [Aircraft, pascal_paintings, mini_ImageNet, meta_iNat, ChestX, tiered_ImageNet, CUB_fewshot, 
                     tiered_meta_iNat, cifar, paintings, pascal]
"""

if __name__ == '__main__':
  way = 5
  args = train_parser()

  strict = forward_method = False
  if 'Finetune' in args.model:
    forward_method = True

  fe_metadata = get_fe_metadata(args)
  test_path = get_test_path(args, data_path)
  device = torch.device(f'cuda:{args.gpu}')

  # Get the model and load its weights.
  
  model, model_path = get_model(args, fe_metadata, device)
  if model_path:
    model.load_state_dict(torch.load(model_path, map_location=f'cuda:{args.gpu}'), strict=strict)
  model.to(device)
  model.eval()

  # Sostituisci il modulo feature_extractor con WikiArt-Style da huggingface
  model_name = "prithivMLmods/WikiArt-Style"  # Sostituisci con il tuo percorso modello
  # Carica il modello base SiglipVisionModel
  siglip_vision_model = SiglipVisionModel.from_pretrained(model_name).to(device)

  # Crea un modulo wrapper per estrarre le feature corrette
  class SiglipFeatureExtractorWrapper(nn.Module):
      def __init__(self, siglip_model):
          super().__init__()
          self.model = siglip_model
          # Assicurati che il modello sottostante sia in modalità eval
          self.model.eval()

      def forward(self, pixel_values):
          # L'input 'inp' da CAML.get_feature_vector dovrebbe essere pixel_values
          # Esegui il modello base Siglip
          outputs = self.model(pixel_values=pixel_values)
          # Restituisci l'output pooled (tensore delle feature)
          return outputs.pooler_output

  # Sostituisci il feature_extractor con il wrapper
  model.feature_extractor = SiglipFeatureExtractorWrapper(siglip_vision_model)

  with torch.no_grad():
    for shot in [5, 1]:
      mean, interval = meta_test(
        data_path=test_path,
        model=model,
        way=way,
        shot=shot,
        pre=False,
        transform_type=fe_metadata['test_transform'],
        trial=10, # NOTA: trial=10 è molto basso per una valutazione affidabile, considera di aumentarlo
        use_forward_method=forward_method)
      print('%d-way-%d-shot acc: %.3f\t%.3f' % (way, shot, mean, interval))

print(model)