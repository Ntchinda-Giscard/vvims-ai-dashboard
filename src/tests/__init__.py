class Compteur:
    objet_cree = 0
    def __init__(self):
        Compteur.objet_cree += 1


if __name__ == "__main__":
    a = Compteur()
    print(f"Compteur {a.objet_cree} ")
    b = Compteur()
    print(f"Compteur {b.objet_cree} ")