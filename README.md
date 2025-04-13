# Text-to-Music and Image Generation Web App

This project is a lightweight web-based interface that bridges users with two generative AI models:

- A **Stable Diffusion** model for **text-to-image** synthesis
- A **Transformer-based** model for **text-to-music** (MIDI) generation

Both models were fine-tuned for the task and are served via a FastAPI backend. Users interact with the system through an intuitive and minimalistic interface that allows inputting natural language prompts or answering a short survey, and receiving media content in return.

---

## Web Interface Design

The FastAPI-powered web app serves four core pages:

### Landing Page

- Introduces the project and its objectives
- Offers navigation to all other functional pages

### Image Generation Page

- Users input a natural language prompt
- The backend sends the prompt to a **Stable Diffusion model**
- Generated image is returned and displayed on the page

### Music Generation Page

- Users input a textual prompt for a musical theme
- The backend generates a **MIDI file** using a text-to-music model
- Users can **listen** via an embedded player or **download** the file

### Survey-Based Generation Page

- Users answer a short survey (multiple-choice)
- Server constructs two prompts: one for image and one for music
- Both models are invoked; outputs are returned for **simultaneous viewing and listening**

All inference is executed **server-side**. The frontend is kept deliberately clean to emphasize generative results.

---

## Models

Pretrained and fine-tuned model weights are **not included in the repo** due to size constraints.

You can download them here:
📁 [Google Drive - Models Folder](https://drive.google.com/drive/folders/1pGIbqd8jr9m9B3IemBjXSfkAr4Czsvbe?usp=sharing)

Place all downloaded files into a `static/` directory within the project root.

---


## Run the App

`uvicorn main:app --reload`

Visit the app in your browser at:

`http://localhost:8000`
