# 🤖 Reinforcement Learning - Semesteraufgabe für Anfänger

**Erstellt für:** Python-Anfänger mit 4-5 Monaten Erfahrung  
**Dauer:** 4 Wochen à 90 Minuten Präsenz  
**Schwierigkeitsgrad:** ⭐⭐⭐ (mittelmäßig)

---

## 📋 Überblick

Diese Semesteraufgabe lehrt Reinforcement Learning durch ein **praktisches Projekt**: 
Trainiere einen KI-Agent, der in einer virtuellen Welt 🌍 Ressourcen sammelt!

### Was ihr lernt
- ✅ Wie KI-Agenten Entscheidungen treffen
- - ✅ Reinforcement Learning & Q-Learning
  - - ✅ Neuronale Netze (Deep Q-Networks) trainieren
    - - ✅ Machine Learning Experimente auswerten
     
      - ---

      ## 🎯 Projektstruktur

      ```
      EDUCATIONAL_PACKAGE/
      ├── 1_Student_Notebook.ipynb          # Hauptaufgabe (mit Lücken zum ausfüllen)
      ├── 2_Solution_Notebook.ipynb         # Tutoren-Lösungen
      ├── 3_WEEK_BY_WEEK_GUIDE.md          # Detaillierter Wochenplan
      ├── requirements.txt                  # Python-Dependencies
      ├── helpers.py                        # Hilfsfunktionen für Visualisierung
      └── README.md                        # Diese Datei
      ```

      ---

      ## 📅 4-Wochen Plan

      ### **WOCHE 1: Foundation** (90 Min)
      - Umgebung verstehen (GridWorld)
      - - Agent-Klasse implementieren
        - - Einfache Navigations-Agenten bauen
          - - **Hausaufgabe:** Random Episodes analysieren
           
            - **Deadline:** Durchschnittliche Scores von 50 Random-Episodes berechnen
           
            - ### **WOCHE 2: Q-Learning** (90 Min)
            - - Q-Learning Algorithmus verstehen
              - - Q-Tabelle implementieren
                - - Diskrete State-Representation
                  - - Trainingslauf & Visualisierung
                    - - **Hausaufgabe:** Q-Learning Parameter tunen
                     
                      - **Deadline:** Trainierten Q-Learning Agent mit Score > 5.0
                     
                      - ### **WOCHE 3: Deep Q-Learning (DQN)** (90 Min)
                      - - Neuronale Netze & CNN verstehen
                        - - DQN-Agent implementieren
                          - - Replay Memory & Target Network
                            - - Trainings-Loop
                              - - **Hausaufgabe:** DQN trainieren mit verschiedenen Learning Rates
                               
                                - **Deadline:** DQN-Modell speichern (.pth Datei)
                               
                                - ### **WOCHE 4: Test & Visualisierung** (90 Min)
                                - - Trainierten Agent in Aktion testen
                                  - - Lernkurven visualisieren
                                    - - Vergleich: Random vs Q-Learning vs DQN
                                      - - Abschlusspräsentation vorbereiten
                                       
                                        - **Final Goal:** Final score > 20.0
                                       
                                        - ---

                                        ## 🚀 Quick Start

                                        ### Installation
                                        ```bash
                                        # 1. Repository klonen
                                        git clone https://github.com/ZeroDTE/reinforcement-learning-abm-.git
                                        cd reinforcement-learning-abm-/EDUCATIONAL_PACKAGE

                                        # 2. Dependencies installieren
                                        pip install -r requirements.txt

                                        # 3. Jupyter starten
                                        jupyter notebook

                                        # 4. Öffne: 1_Student_Notebook.ipynb
                                        ```

                                        ### Erste Schritte
                                        1. Öffne `1_Student_Notebook.ipynb` in Jupyter
                                        2. 2. Lese die Einführung in **Phase 1**
                                           3. 3. Starte mit dem Code und fülle die `TODO`-Blöcke aus
                                              4. 4. Am Ende jeder Phase gibt es Hausaufgaben
                                                
                                                 5. ---
                                                
                                                 6. ## 📚 Tutoren-Guide
                                                
                                                 7. Für Tutoren gibt es `2_Solution_Notebook.ipynb` mit:
                                                 8. - Alle Lösungen zu den TODOs
                                                    - - Erklärungen hinter jedem Schritt
                                                      - - Zusätzliche Insights und Best Practices
                                                        - - Diskussionsfragen für die Präsenztermine
                                                         
                                                          - ### Tipps für Präsenztermine
                                                          - - **Woche 1:** Visualisiert das GridWorld-Environment zusammen
                                                            - - **Woche 2:** Debuggt Q-Learning gemeinsam (warum lernt der Agent?)
                                                              - - **Woche 3:** Erklärt CNN-Layer und warum sie besser sind
                                                                - - **Woche 4:** Analysiert Lernkurven und diskutiert Verbesserungen
                                                                 
                                                                  - ---

                                                                  ## 🛠️ Anforderungen

                                                                  - **Python:** 3.8+
                                                                  - - **Hauptbibliotheken:** NumPy, PyTorch, Matplotlib
                                                                    - - **Zeit:** ~10-15 Stunden total (3-4 Std/Woche Hausaufgaben)
                                                                      - - **GPU:** Optional (CPU funktioniert auch, dauert länger)
                                                                       
                                                                        - ---

                                                                        ## 📊 Erwartete Ergebnisse

                                                                        Nach 4 Wochen sollten die Studierenden folgende Scores erreichen:

                                                                        | Agent | Woche 1 | Woche 2 | Woche 3 | Woche 4 |
                                                                        |-------|---------|---------|---------|---------|
                                                                        | Random | 2-3 | - | - | - |
                                                                        | Q-Learning | - | 5-8 | - | - |
                                                                        | DQN | - | - | 15-25 | 20-30 |

                                                                        ---

                                                                        ## 🤔 FAQ

                                                                        **F: Muss ich GPU haben?**
                                                                        A: Nein, CPU funktioniert auch. GPU ist ~5x schneller aber nicht notwendig.

                                                                        **F: Wie lange dauert das Training?**
                                                                        A: Q-Learning: ~5 Min | DQN: ~20 Min (CPU) | DQN: ~5 Min (GPU)

                                                                        **F: Was ist ein "State"?**
                                                                        A: Was der Agent aktuell sieht. Woche 1 erklärt das Grundkonzept.

                                                                        **F: Kann ich mein eigenes Environment bauen?**
                                                                        A: Ja! Nach Woche 3 seid ihr ready für Experimente.

                                                                        ---

                                                                        ## 📖 Zusätzliche Ressourcen

                                                                        - [Sutton & Barto: Reinforcement Learning (Buch)](http://incompleteideas.net/book/the-book-2nd.html)
                                                                        - - [Deep Reinforcement Learning (DeepMind Blog)](https://deepmind.com/blog/article/deep-reinforcement-learning)
                                                                          - - [PyTorch Tutorial](https://pytorch.org/tutorials/)
                                                                           
                                                                            - ---

                                                                            ## 🤝 Fragen/Probleme?

                                                                            Falls es Fragen gibt:
                                                                            1. Schau ins `WEEK_BY_WEEK_GUIDE.md`
                                                                            2. 2. Schau in `2_Solution_Notebook.ipynb`
                                                                               3. 3. Frag auf Discord/Slack der Vorlesung
                                                                                  4. 4. GitHub Issues öffnen
                                                                                    
                                                                                     5. ---
                                                                                    
                                                                                     6. ## 📄 Lizenz
                                                                                    
                                                                                     7. Diese Semesteraufgabe basiert auf dem Original-Projekt von [ZeroDTE](https://github.com/ZeroDTE).
                                                                                     8. Vereinfacht und angepasst für Unterricht.
                                                                                    
                                                                                     9. ---
                                                                                    
                                                                                     10. **Viel Spaß beim Lernen! 🚀**
