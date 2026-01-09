# Woche-für-Woche Anleitung

## Woche 1: Foundation

Ziel: Verstehe die GridWorld Umgebung und implementiere einfache Agenten.

### Was ihr macht
- Die GridWorld Klasse analysieren
- - reset() und step() verstehen
  - - Kleine Experimente mit dem Environment
   
    - ### Hausaufgabe
    - Schreibt eine Funktion random_episode(), die eine Episode mit zufälligen Aktionen spielt und den Score zurückgibt. Führt 50 solche Episodes aus und berechnet Durchschnitt, Min, Max.
   
    - Expected result: Score sollte bei 0-5 liegen.
   
    - ---

    ## Woche 2: Q-Learning

    Ziel: Trainiert einen Q-Learning Agent der lernt.

    ### Was ihr macht
    - Q-Learning Theorie: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max Q(s') - Q(s,a))
    - - State Diskretisierung implementieren
      - - Q-Tabelle aufbauen
        - - Agent trainieren
         
          - ### Hausaufgabe
          - Testet verschiedene Hyperparameter (learning_rate, discount_factor). Was funktioniert besser?
         
          - Expected result: Agent mit Score > 5.
         
          - ---

          ## Woche 3: Deep Q-Learning (DQN)

          Ziel: Mit neuronalen Netzen trainieren.

          ### Was ihr macht
          - CNN Architektur aufbauen
          - - DQN Agent implementieren
            - - Replay Memory verstehen
              - - Erstes Training starten
               
                - ### Hausaufgabe
                - Trainiert einen DQN mit verschiedenen Learning Rates. Vergleicht die Lernkurven.
               
                - Expected result: Modell speichern als .pth Datei.
               
                - ---

                ## Woche 4: Test und Abschluss

                Ziel: Agenten testen und Ergebnisse vergleichen.

                ### Was ihr macht
                - Trainierten Agent laden
                - - 20 Test Episodes spielen
                  - - Lernkurven plotten
                    - - Vergleich: Random vs Q-Learning vs DQN
                     
                      - ### Finale Aufgabe
                      - Erstellt eine Tabelle mit allen Ergebnissen:
                      - - Agent Name
                        - - Average Score
                          - - Max Score
                            - - Training Zeit
                             
                              - ---

                              ## Tipps für Tutoren

                              Vor jedem Termin:
                              - Code selbst testen
                              - - Dependencies prüfen
                                - - Demo vorbereiten
                                 
                                  - Während des Termins:
                                  - - Langsam erklären, RL ist komplex
                                    - - Code live debuggen zeigen
                                      - - Fragen stellen lassen
                                       
                                        - Nach dem Termin:
                                        - - Feedback zu Hausaufgaben geben
                                          - - Probleme notieren
                                           
                                            - ### Common Issues
                                           
                                            - Problem: Agent lernt nicht (Score bleibt bei -0.1)
                                            - Grund: Zu viele Wände oder schlechte Parameter
                                            - Loesung: Weniger Hindernisse testen oder Learning Rate erhöhen
                                           
                                            - Problem: DQN Training dauert lange
                                            - Grund: Nur CPU
                                            - Loesung: Das ist normal, GPU wäre schneller
                                           
                                            - Problem: Lernkurve sehr zittrig
                                            - Grund: Batch Size zu klein
                                            - Loesung: BATCH_SIZE erhöhen
                                           
                                            - ---

                                            ## Was die Studenten lernen

                                            Nach 4 Wochen können sie:
                                            - State, Action, Reward verstehen
                                            - - Q-Learning von Hand durchrechnen
                                              - - PyTorch Code schreiben
                                                - - Neuronale Netze trainieren
                                                  - - Lernkurven interpretieren
                                                    - 
