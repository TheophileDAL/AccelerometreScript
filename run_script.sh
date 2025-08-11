#!/bin/bash
read -p "Nom du fichier : " filename
read -p "Fréquence (Hz) : " freq
chemin="/Users/theophiledal/TierceProjet/Benoit/accelerometre/"
/usr/local/bin/python3.11 "${chemin}main.py" "${chemin}${filename}" "$freq"