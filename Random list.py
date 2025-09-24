import random

def llista_aleatoria_vertical(inici, fi):
  if inici > fi:
    print("L'interval no és vàlid (el número d'inici ha de ser menor o igual que el número final).")
    return

  llista_ordenada = list(range(inici, fi + 1))
  random.shuffle(llista_ordenada)

  # Imprimir la llista verticalment
  for numero in llista_ordenada:
    print(numero)

# Crear i imprimir la llista aleatòria vertical amb els números del [10, 109]
llista_aleatoria_vertical(10, 109)
