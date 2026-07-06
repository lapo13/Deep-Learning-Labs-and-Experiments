from semantic_indexer import semantic_indexer
from wrapper_CLIP import wrapper_CLIP
from wrapper_translator import wrapper_translator
from datasets import load_dataset, concatenate_datasets, load_from_disk
import gradio as gr
import os, json, time, signal, threading
from datetime import datetime


## Parametri globali e setup Iniziale ##
model_name_translation = "Qwen/Qwen2.5-0.5B-Instruct"
image_model_name = "openai/clip-vit-base-patch16"
app_data_path = "./AppData"
data_oggi = datetime.now().strftime("%Y-%m-%d")

os.makedirs(app_data_path, exist_ok=True)

## Definizione delle funzioni principali dell'applicazione ##

def salva_nota_log(input_utente, prompt_en, nota, file_log=app_data_path+ "/note/registro_ricerche.jsonl"):
    os.makedirs(os.path.dirname(file_log), exist_ok=True)
    # Se il box è vuoto, non fare nulla
    if not nota.strip():
        return gr.update(placeholder="Scrivi qualcosa prima di salvare!")
        
    ora_attuale = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Strutturiamo il JSON identificando questo record come una "nota"
    log_nota = {
        "data_ora": ora_attuale,
        "tipo_evento": "nota_utente",
        "ricerca_corrente": {
            "input_utente": input_utente,
            "prompt_en": prompt_en
        },
        "nota": nota.strip()
    }
    
    # Scrittura in coda nel file unico JSONL
    with open(file_log, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_nota, ensure_ascii=False) + "\n")
        
    print(f"Nota salvata nel registro per la ricerca: '{input_utente}'")
    
    # Svuota il box delle note e aggiorna il placeholder per dare un feedback visivo all'utente
    return gr.update(value="", placeholder="Nota salvata con successo!")

def timer_autodistruzione(ore=12):
    secondi_attesa = ore * 60 * 60
    
    print(f"Timer di auto-spegnimento avviato: l'applicazione si chiuderà tra {ore} ore.")
    time.sleep(secondi_attesa)
    
    print("Tempo scaduto! Chiusura automatica dell'applicazione in corso...")
    
    os.kill(os.getpid(), signal.SIGINT)

def salva_log(input_utente, prompt_en, indici_immagini, esito="Successo", cartella_logs="logs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_log = os.path.join(cartella_logs, f"ricerca_{timestamp}.json")
    
    log_data = {
        "data_ora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ricerca": {
            "input_utente": input_utente,
            "prompt_en": prompt_en,
            "indici_immagini": indici_immagini,
            "esito": esito
        }
    }
    
    os.makedirs(cartella_logs, exist_ok=True)
    with open(file_log, mode="w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=5)

def search_database(query_testo, numero_risultati, logging_path):
    prompt_originale = query_testo
    
    try:
        prompt_inglese = model_translation.translate(query_testo)
        indici = semantic_index.search(prompt_inglese, top_k=numero_risultati)
        
        risultati = [(ds['image'][i], ds['caption_0'][i]) for i in indici]

        salva_log(
            input_utente=prompt_originale,   
            prompt_en=prompt_inglese, 
            indici_immagini=indici,
            esito=f"Successo - Mostrati {len(risultati)} risultati", 
            cartella_logs=logging_path
        )
        
        return prompt_originale, prompt_inglese, risultati

    except Exception as e:
        try:
            salva_log(
                input_utente=query_testo, 
                prompt_en="****", 
                indici_immagini="****", # Sarà vuota o parziale
                esito=f"Errore: {str(e)}",
                cartella_logs=logging_path
            )
        except Exception as log_err:
            print(f"Impossibile scrivere il file di log JSON: {log_err}")
            
        return query_testo, f"Errore di sistema: {str(e)}", []

def cambia_slide(risultati, indice_attuale, direzione):
        nuovo_indice = indice_attuale + direzione
        
        # Estrai la nuova immagine e didascalia usando il nuovo indice numerico
        img, cap = risultati[nuovo_indice]
        testo_info = f"<h3 style='text-align: center;'>{cap}</h3><p style='text-align: center; color: gray;'>Risultato {nuovo_indice + 1} di {len(risultati)}</p>"
        
        # Controllo dei bottoni: spegne "Precedente" se sei a 0, spegne "Successivo" se sei alla fine
        attiva_prev = nuovo_indice > 0
        attiva_next = nuovo_indice < (len(risultati) - 1)
        
        return (
            nuovo_indice, 
            img, 
            testo_info, 
            gr.update(interactive=attiva_prev), 
            gr.update(interactive=attiva_next)
        )

def inizializza_slideshow(testo, top_k):
        p_it, p_en, risultati = search_database(testo, top_k, logging_path=app_data_path+f"/registro_ricerche_{data_oggi}")
        
        if not risultati or len(risultati) == 0:
            # Caso in cui non ci sono risultati
            return p_it, p_en, [], 0, None, "<h3 style='text-align: center; color: red;'>Nessun risultato trovato.</h3>", gr.update(interactive=False), gr.update(interactive=False)
            
        # Carica il primo risultato (Indice 0)
        prima_img, prima_cap = risultati[0]
        testo_info = f"<h3 style='text-align: center;'>{prima_cap}</h3><p style='text-align: center; color: gray;'>Risultato 1 di {len(risultati)}</p>"
        
        # Sblocca il tasto "Successivo" solo se ci sono effettivamente più di 1 immagine
        puo_andare_avanti = len(risultati) > 1
        
        return (
            p_it, 
            p_en, 
            risultati,            # Salva i dati nello stato risultati_memoria
            0,                    # Resetta l'indice_corrente a 0
            prima_img,            # Aggiorna lo schermo gr.Image
            testo_info,           # Aggiorna la didascalia
            gr.update(interactive=False),               # Precedente disattivato (siamo alla prima foto)
            gr.update(interactive=puo_andare_avanti)    # Successivo attivo/disattivo
        )
    

## Caricamento del dataset e inizializzazione dei modelli ##
ds_save_path = app_data_path + "/flickr8k_dataset"
if not os.path.exists(ds_save_path):
    ds_train = load_dataset("jxie/flickr8k")['train']. remove_columns(['caption_1', 'caption_2', 'caption_3', 'caption_4'])
    ds_eval = load_dataset("jxie/flickr8k")['validation']. remove_columns(['caption_1', 'caption_2', 'caption_3', 'caption_4'])
    ds_test = load_dataset("jxie/flickr8k")['test']. remove_columns(['caption_1', 'caption_2', 'caption_3', 'caption_4'])
    ds = concatenate_datasets([ds_train, ds_eval, ds_test])
    ds.save_to_disk(ds_save_path)
else:
    ds = load_from_disk(ds_save_path)

model = wrapper_CLIP(model_name= image_model_name, model_save_path=app_data_path + "/clip_model")
model_translation = wrapper_translator(model_name=model_name_translation, model_save_path=app_data_path + "/translation_model")
semantic_index= semantic_indexer(model=model, index_embeddings_path=app_data_path + "/index_embeddings.pt")
semantic_index.build_index(ds['image'], batch_size=8)

with gr.Blocks(title="Motore CLIP", theme=gr.themes.Soft()) as interfaccia: # type: ignore
    risultati_memoria = gr.State([])
    indice_corrente = gr.State(0)
    
    # Intestazione
    gr.HTML("<h1 style='text-align: center; color: #ff7c00;'>Motore di Ricerca Semantica Multilingua</h1>")
    gr.HTML("<p style='text-align: center; font-size: 16px;'>I risultati verranno mostrati in una slideshow interattiva con didascalia.</p>")
    
    # 2. BARRA DI RICERCA
    with gr.Row(variant="panel", equal_height=True):
        with gr.Column(scale=4):
            input_testo = gr.Textbox(
                placeholder="Es: un cane nero che corre nel prato...",
                show_label=False,
                lines=1,
                container=False, 
            )
        with gr.Column(scale=1, min_width=150):
            input_slider = gr.Slider(
                minimum=2, maximum=10, step=1, value=5,
                label="N° Risultati",
                container=False
            )
        with gr.Column(scale=1, min_width=100):
            btn_cerca = gr.Button("Cerca", variant="primary", size="lg")
            
    # 3. AREA TRADUZIONI 
    with gr.Row():
        output_prompt_it = gr.Textbox(label="Prompt originale (IT)", interactive=False)
        output_prompt_en = gr.Textbox(label="Prompt tradotto per CLIP (EN)", interactive=False)
        
    # 4. GRANDE AREA SLIDESHOW
    with gr.Column(variant="panel"):
        output_immagine = gr.Image(
            label="Visualizzatore", 
            interactive=False, 
            height=450,
            show_label=False
        )
        
        output_didascalia = gr.Markdown(
            "<h3 style='text-align: center; color: gray;'>Fai una ricerca per iniziare...</h3>"
        )
        
        # Pulsanti di navigazione orizzontali
        with gr.Row():
            btn_prev = gr.Button("Precedente", interactive=False)
            btn_next = gr.Button("Successivo", interactive=False)
            
        with gr.Row():
            input_note = gr.Textbox(
                label="Aggiungi una nota o un feedback su questa ricerca",
                placeholder="Es: 'La foto 3 è un falso positivo', 'Traduzione eccellente'...",
                lines=2,
                max_lines=4
            )
        with gr.Row():
            btn_salva_nota = gr.Button("Salva Nota / Invia Feedback", variant="secondary")


    # --- COLLEGAMENTO DELLE AZIONI AI COMPONENTI ---

    # Collegamento tasto Cerca e tasto Invio sulla tastiera
    for evento in [btn_cerca.click, input_testo.submit]:
        evento(
            fn=inizializza_slideshow,
            inputs=[input_testo, input_slider],
            outputs=[output_prompt_it, output_prompt_en, risultati_memoria, indice_corrente, output_immagine, output_didascalia, btn_prev, btn_next]
        )

    # Collegamento pulsante Precedente
    btn_prev.click(
        fn=lambda r, i: cambia_slide(r, i, -1),
        inputs=[risultati_memoria, indice_corrente],
        outputs=[indice_corrente, output_immagine, output_didascalia, btn_prev, btn_next]
    )
    
    # Collegamento pulsante Successivo
    btn_next.click(
        fn=lambda r, i: cambia_slide(r, i, 1),
        inputs=[risultati_memoria, indice_corrente],
        outputs=[indice_corrente, output_immagine, output_didascalia, btn_prev, btn_next]
    )

    btn_salva_nota.click(
        fn=salva_nota_log,
        inputs=[input_testo, output_prompt_en, input_note],
        outputs=[input_note]
    )

# Avvia l'applicazione locale
if __name__ == "__main__":
    thread_timer = threading.Thread(target=timer_autodistruzione, args=(6,), daemon=True)
    thread_timer.start()
    
    interfaccia.launch(share=True)
