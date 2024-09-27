from transformers import BertModel

from cenai_core import load_dotenv


load_dotenv()

model = BertModel.from_pretrained("bert-base-cased")