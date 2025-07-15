import os
import numpy as np
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from milvus_connect import Milvus_Connect

# Load local model and tokenizer
model_path = "BERT" # your local model path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
model = model.half().cuda()

# Create a Milvus collection
def create_collection(collection_name):
    fields = [
        FieldSchema(name="english_word", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="english_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="chinese_word", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
        FieldSchema(name="chinese_vector", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields, description="Geology vocabulary collection")
    collection = Collection(name=collection_name, schema=schema)
    return collection

#  Read Chinese and English vocabulary from the file
def load_vocab_from_files(directory):
    chinese_words = []
    english_words = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.txt'):
            with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as file:
                for line in file:
                    if line.startswith("english："):
                        english_word = line.replace("english：", "").strip()
                        english_words.append(english_word)
                    elif line.startswith("chinese："):
                        chinese_word = line.replace("chinese：", "").strip()
                        chinese_word = chinese_word.split('(')[0].strip()

                        # Make sure Chinese words do not exceed the maximum length of 200 characters
                        if len(chinese_word) > 200:
                            print(f"Warning: '{chinese_word}' is too long and will be trimmed.")
                            chinese_word = chinese_word[:100]

                        chinese_words.append(chinese_word)
    return english_words, chinese_words

# Vectorized encoding
def get_vector(text):
    inputs = tokenizer(text, return_tensors='pt').to("cuda")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().astype(np.float32)

# Store in Milvus database, support batch insertion
def insert_to_milvus(collection, english_words, chinese_words, english_vectors, chinese_vectors, batch_size=1000):
    num_entries = len(english_words)
    num_batches = (num_entries + batch_size - 1) // batch_size
    for i in tqdm(range(0, num_entries, batch_size), total=num_batches, desc="Inserting batches"):
        end_index = min(i + batch_size, num_entries)
        batch_english_words = english_words[i:end_index]
        batch_chinese_words = chinese_words[i:end_index]
        batch_english_vectors = english_vectors[i:end_index]
        batch_chinese_vectors = chinese_vectors[i:end_index]

        # Make sure the vector is a 2D array
        batch_english_vectors = np.vstack(batch_english_vectors)
        batch_chinese_vectors = np.vstack(batch_chinese_vectors)

        entities = [
            batch_english_words,
            batch_english_vectors.tolist(),
            batch_chinese_words,
            batch_chinese_vectors.tolist()
        ]
        collection.insert(entities)

        # Print batch information
        print(f"Inserted batch {i // batch_size + 1}: {len(batch_english_words)} entries (from index {i} to {end_index - 1})")

    # Create English vector index
    collection.create_index(field_name="english_vector",
                            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}})

    # Create Chinese vector index
    collection.create_index(field_name="chinese_vector",
                            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}})


    collection.load()

    # Verify that the inserted vector is valid
    verify_vectors(collection, english_words, chinese_words)


def verify_vectors(collection, english_words, chinese_words):
    for english_word in english_words:

        result = collection.query(expr=f'english_word == "{english_word}"', output_fields=["english_vector"])

        if result and len(result) > 0:
            vector = result[0]["english_vector"]
            # Check if the vector is valid
            if isinstance(vector, list) and len(vector) == 768:
                print(f"Verification successful: The vector for the English word '{english_word}' is valid.")
            else:
                print(f"Validation failed: Invalid vector for English word '{english_word}'.")

        else:
            print(f"Verification failed: English word '{english_word}' not found.")

    for chinese_word in chinese_words:

        result = collection.query(expr=f'chinese_word == "{chinese_word}"', output_fields=["chinese_vector"])

        if result and len(result) > 0:
            vector = result[0]["chinese_vector"]

            if isinstance(vector, list) and len(vector) == 768:
                print(f"Verification successful: The vector of the Chinese word '{chinese_word}' is valid.")
            else:
                print(f"Verification failed: The vector for the Chinese word '{chinese_word}' is invalid.")
        else:
            print(f"Verification failed: Chinese word '{chinese_word}' not found.")


def main():

    collection_name = "Dictdata_collection"

    # Connect to Milvus
    milvus_connection = Milvus_Connect()

    # Ensure the connection is successful before creating the collection
    connections.connect("default", uri="http://127.0.0.1:19530")
    collection = create_collection(collection_name)

    # Specify the data directory
    text_directory ="dictdata"  # Replace with your txt folder path


    english_words, chinese_words = load_vocab_from_files(text_directory)


    english_vectors = [get_vector(word) for word in tqdm(english_words, desc="Generating English vectors")]
    chinese_vectors = [get_vector(word) for word in tqdm(chinese_words, desc="Generating Chinese vectors")]

    # Insert data into Milvus
    insert_to_milvus(collection, english_words, chinese_words, english_vectors, chinese_vectors)
    print("Insert data successfully done!")

#——————————————————————————————————————————
    # Get the collection schema
    schema = collection.schema

    # Print schema to view field types
    for field in schema.fields:
        print(f"Field name: {field.name}, Field type: {field.dtype}")
#————————————————————————————————————————————————————
    # Verify data
    result = collection.query(expr="chinese_word in ['浊沸石']", output_fields=["chinese_word", "english_word", "chinese_vector"])
    #print(result)

if __name__ == "__main__":
    main()
