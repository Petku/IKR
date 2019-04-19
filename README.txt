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
Návod na jeho inštaláciu a použitie je možno nájsť na tejto adrese:

https://docs.python-guide.org/dev/virtualenvs/?fbclid=IwAR0AQKG4V74v32bVtJk9VYnh3SiYTm9QpntTPUArxuQoMA0wuTLTJ9WFCWs


Použité knižnice:
-----------------
sklearn
numpy
__future__
time
ikrlib

Využili sme knižnicu ikrlib. Konkrétne funkcie na načítanie png, waw súborov (png2fea - túto funkciu sme rozširili 
o metódu flatten(), wav16khz2mfcc) logpdf_gmm a train_gmm.


Spustenie skriptov:
-------------------
./python2.7 image_neural_network.py
./python2.7 audio.py


Získanie výsledkov:
-------------------
Výsledky identifikácie možno získať spustením jednotlivých skriptov, ktoré ich vypíšu na //TODO stdout/súboru.


Použita verzia Pythonu: 2.7

 
