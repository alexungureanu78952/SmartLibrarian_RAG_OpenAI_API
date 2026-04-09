## Plan: Chatbot RAG Cărți (CLI)

Construim întâi un MVP robust pe CLI cu OpenAI GPT + ChromaDB + tool calling pentru rezumat complet după recomandare, apoi lăsăm extensibil pentru opționale. Fluxul recomandat: ingestie rezumate -> indexare embeddings -> retrieval semantic -> recomandare LLM -> apel automat tool get_summary_by_title -> răspuns final + opțional TTS. Include și filtru simplu de limbaj nepotrivit înainte de apelul LLM.

**Steps**
1. Faza 1: Scaffold proiect Python și configurare mediu
2. Definește structura minimă de proiect (pachet sursă, date, scripturi rulare, teste de bază) și configurează variabilele de mediu pentru cheile API. *(blocant pentru toți pașii următori)*
3. Pin-uiește dependențele pentru reproductibilitate (OpenAI SDK, ChromaDB, utilitare CLI, TTS opțional) și documentează exact comenzile de instalare/rulare.
4. Faza 2: Date + indexare RAG
5. Creează fișierul book_summaries cu minim 10 cărți, rezumate 3-5 rânduri, teme clare (English-only conform deciziei) și titluri normalizate.
6. Implementează pipeline-ul de ingestie: parsează rezumatele în documente, adaugă metadate (title, themes/source), generează embeddings cu modelul ales, persistă în ChromaDB local. *(depinde de 1-3)*
7. Configurează retriever semantic (top_k, optional threshold) și adaugă un smoke-check care validează că interogări tematice returnează cărți relevante. *(depinde de 6)*
8. Faza 3: Chat + tool calling
9. Implementează get_summary_by_title(title: str) pe sursă locală (dict/JSON) cu matching exact case-insensitive și fallback prietenos pentru titlu inexistent.
10. Înregistrează funcția ca tool în OpenAI Chat API și construiește bucla de tool-calling: modelul recomandă titlul, apoi aplicația apelează automat tool-ul pentru rezumatul complet, apoi compune răspunsul final. *(depinde de 7 și 9)*
11. Adaugă interfața CLI conversațională (input loop + exit commands + mesaje ghidaj) care folosește același orchestrator RAG + tool.
12. Faza 4: Guardrails + opțional TTS
13. Adaugă filtru simplu de limbaj nepotrivit înainte de retriever/LLM: dacă detectează termeni din listă locală, răspunde politicos fără apel model. *(depinde de 11)*
14. Faza 5: Documentație + validare finală
15. Scrie README complet: arhitectură, setup, indexare, rulare chat, exemple întrebări, limitări și troubleshooting.
16. Include exemple obligatorii de testare și scenarii edge-case (titlu lipsă, query fără context, input ofensator).

**Relevant files**
- /home/alexungureanu/demopython/devbox.json — păstrat pentru context de mediu (uv deja disponibil)
- /home/alexungureanu/demopython/pyproject.toml — dependențe și scripturi
- /home/alexungureanu/demopython/README.md — pași build/rulare + flow explicat
- /home/alexungureanu/demopython/.env.example — variabile necesare (OPENAI_API_KEY etc.)
- /home/alexungureanu/demopython/data/book_summaries.md — corpus minim 10+ cărți
- /home/alexungureanu/demopython/data/book_summaries.json — sursă structurată pentru tool
- /home/alexungureanu/demopython/src/ragbot/indexer.py — ingestie + embeddings + persistență Chroma
- /home/alexungureanu/demopython/src/ragbot/retriever.py — căutare semantică
- /home/alexungureanu/demopython/src/ragbot/tools.py — get_summary_by_title + schema tool
- /home/alexungureanu/demopython/src/ragbot/chat.py — orchestration LLM + tool-calling
- /home/alexungureanu/demopython/src/ragbot/ui_cli.py — interfață CLI
- /home/alexungureanu/demopython/src/ragbot/safety.py — filtru limbaj nepotrivit
- /home/alexungureanu/demopython/src/ragbot/tts.py — generare audio (opțional)
- /home/alexungureanu/demopython/tests/test_tools.py — teste tool summary
- /home/alexungureanu/demopython/tests/test_retrieval_smoke.py — verificare retrieval minimă

**Verification**
1. Rulează indexarea end-to-end și verifică existența colecției Chroma + numărul de documente indexate.
2. Run queries such as "friendship and magic", "freedom and social control", and "war stories" and validate top-result relevance.
3. Run query "What is 1984 about?" and verify tool-calling: recommendation + full summary from get_summary_by_title.
4. Rulează query cu titlu inexistent și verifică fallback fără crash.
5. Rulează input ofensator și confirmă că răspunsul vine din filtru local fără apel LLM.
6. Dacă TTS este activ, verifică generarea fișierului audio și conținutul text-audio corespunzător răspunsului final.
7. Rulează set minim de teste automate pentru tool și retrieval smoke.

**Decisions**
- Interface: CLI first, English-only messages and prompts.
- Scope inițial: Core (RAG + tool calling) + filtru limbaj local (fără TTS).
- Nivel: MVP rapid, nu implementare full web frontend în această etapă.
- Limbă conținut: English-only în datele de rezumate pentru retrieval mai stabil.
- Exclus din plan inițial: TTS, STT, image generation, frontend separat (React/Angular/Vue).

**Further Considerations**
1. Model chat recomandat: GPT orientat cost/latency pentru demo; dacă vrei calitate mai mare, crește modelul doar după validarea flow-ului.
2. Parametri retrieval inițiali recomandați: chunk mediu + top_k 3-5, apoi ajustare pe exemplele tale de întrebări.
3. Dacă Chroma ridică probleme de mediu, fallback documentat: alt vector store local cu aceeași interfață retriever.
