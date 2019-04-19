Projekt: 	Identifikácia osôb
Dátum:		19. 04. 2019
Predmet: 	IKR
Autori:		Patrik Nikolas, Zaujec Andrej, Horvát Jozef
======================================================================================================================

Stručný popis:
--------------

Tento projekt je úlohou do predmetu IKR na VUT FIT. Zadaním bolo vytvoriť program na identifikáciu 31 rôznych osôb 
z obrázku tváre a hlasovej nahrávky. 

Riešenie obsahuje skripty napísané v jazyku Python: audio.py a image_neural_network.py umiestnené v zdrojovom adresári.


Pred spustením:
---------------

Pred spustením si odporúčame vytvoriť virtuálne prostredie pomocou nástroja virtualenv. 
V nom potom už len stačí zadat "pip install -r requirements.txt"  a budú nainštalované všetky potrebné knižnice.

Použité knižnice:
-----------------
sklearn - scikit
numpy
time
ikrlib

Využili sme knižnicu ikrlib. Konkrétne funkcie na načítanie png, waw súborov (png2fea - túto funkciu sme rozširili 
o metódu flatten(), wav16khz2mfcc) logpdf_gmm a train_gmm.


Spustenie skriptov:
-------------------
Pri spustení musia byť skripty v rovnakom priečinku ako sú dáta v repozitároch dev/ train/ a eval/
./python2.7 image_neural_network.py
./python2.7 audio.py


Získanie výsledkov:
-------------------
Výsledky identifikácie možno získať spustením jednotlivých skriptov, ktoré ich vypíšu na //TODO stdout/súboru.


Použita verzia Pythonu: 2.7

 
