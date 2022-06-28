# animals-embedding

## Introduction
This is a mini-project of implementing Image Embedding with Animal Dataset using EfficientNetB0 as the backbone.

## Tutorial

Clone the project

```bash
git clone https://github.com/zogojogo/animal-image-embedding-wii.git
```

Go to the project directory

```bash
cd animals-embedding
```

Download Dependencies
```bash
pip install -r requirements.txt
```

Start API service

```
python3 app.py
```
  
## API Reference

Service: http://your-ip-address:8080

#### POST image

```http
  POST /animals_embedding
```
Content-Type: multipart/form-data
| Name    | Type   | Description                                         |
| :------ | :----- | :-------------------------------------------------- |
| `image` | `file` | **Required**. `image/png` MIME Type |

## Output Example

**Output:**<br>
```python
{
  "filename": "<filename>",
  "similar image": "<string>",
  "similarity": "<float>",
  "inference time": "<inference time>"
}
```