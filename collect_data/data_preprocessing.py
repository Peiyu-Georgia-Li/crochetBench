from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import PromptTemplate
from typing import List
from langchain_openai import ChatOpenAI
import json
import os




class Document(BaseModel):
    pattern_name: str = Field(description="Pattern Name")
    materials: List[str] = Field(description="Materials")
    skill_type: str = Field(description="The type of skill required: crochet or knitting")
    skill_level: str = Field(description="Skill Level: beginner, easy, intermediate, or experienced")
    abbreviations: List[str] = Field(description="Abbreviations")
    measurements: str = Field(description="Measurements")
    gauge: str = Field(description="Gauge")
    instructions: str = Field(description="Instructions: includes all rounds or rows in each pages, omit the meta data in each pages")




def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages


# --- API setup ---
# Get API key from environment variable for security
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
parser = OutputFixingParser.from_llm(llm,parser=JsonOutputParser(pydantic_object=Document))

prompt = PromptTemplate(
    template="You are extracting crochet instructions from text. Please output the full JSON structure below. Do not truncate any information. Ensure that the \"instructions\" field contains the complete and uncut pattern, even if it is long. Do not summarize.\n{format_instructions}\n{context}",
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

#####################################################
# test single pdf
# existing_data = []
# address = "./input_file/Afghans+%26+Blankets/ALC0202-025204M.pdf"
# pages = load_pdf(address)
# chain = prompt | llm | parser
# try:
#     response = chain.invoke({
#         "context":pages
#     })
# except json.decoder.JSONDecodeError as e:
#     print(f"Error parsing JSON: {e}")
#     print(address)
# response["id"]=address.split('/')[-1].replace(".pdf", "")
# response["project_type"]="Afghans or Blankets"
# # response["project_type"]=address.split('/')[-2]
# response["source"] = address
# print(response)

#####################################################


# Process all directories in input_file
input_base_dir = "../data/pdf_file/"
output_base_dir = "../data/crochet_pattern_by_project/"

# Ensure output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Directories to exclude from processing
# exclude_dirs = ["Afghans+Blankets", "Hats", "Rugs", "Aprons", "Tops", "Bunting+Bags", "Hair+Accessories", "Coasters", "Baskets", "Wall+Hangings", "Cowls"]

# Add filenames from output directory to exclude_dirs
for filename in os.listdir(output_base_dir):
    if filename.endswith('.json'):
        # Remove the .json extension to get the base filename
        base_name = filename.replace('.json', '')
        exclude_dirs.append(base_name)

print("⚠️exclude_dirs:", exclude_dirs)


# Iterate over all directories in input_file
for dir_name in os.listdir(input_base_dir):
    dir_path = os.path.join(input_base_dir, dir_name)
    
    # Skip if not a directory or if in exclude list
    if not os.path.isdir(dir_path) or dir_name in exclude_dirs:
        continue
        
    print(f"Processing directory: {dir_name}")

        
    # Create json filename based on directory name
    json_filename = dir_name
    json_file_path = os.path.join(output_base_dir, f"{json_filename}.json")
    
    # Load existing data or initialize empty list
    try:
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            existing_data = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    
    print(f"Processing directory: {dir_name}")
    print(f"Saving to: {json_file_path}")
    
    count = 0
    for filename in os.listdir(dir_path):
        if not filename.lower().endswith('.pdf'):
            continue
            
        address = os.path.join(dir_path, filename)
        print(f"Processing file: {address}")
        
        try:
            pages = load_pdf(address)
            chain = prompt | llm | parser
            response = chain.invoke({
                "context": pages
            })
            
            # Check if response is None
            if response is None:
                print(f"Warning: Received None response for {address}, skipping this file")
                continue
                
        except Exception as e:
            print(f"Error processing {address}: {e}")
            continue
            
        # Now we know response is not None
        response["id"] = filename.replace(".pdf", "")
        # Apply the requested replacements to create project_type
        project_type = dir_name.replace("+%26+", " or ").replace("+", " ")
        response["project_type"] = project_type
        response["source"] = address
        
        print(f"Processed: {filename}")
        existing_data.append(response)
        count += 1
        print(f"Files processed in this directory: {count}")
        
    # Save the updated data
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
        
    print(f"Instructions saved to {json_file_path}")

print("All directories processed successfully!")
