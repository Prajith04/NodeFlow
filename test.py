from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import JSONLoader
loader=UnstructuredLoader("/home/prajith/Desktop/flowbit/sample_email.eml")
email=loader.load()
print(email)
loader=PyPDFLoader("/home/prajith/Desktop/flowbit/sample-invoice.pdf")
pdf=loader.load()
print(pdf)
loader=JSONLoader("/home/prajith/Desktop/flowbit/sample_data.json", jq_schema=".[]",)
json=loader.load()
print(json)