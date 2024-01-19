## ECO-ontology

U ovom projektu urađena je segmentacija terena na osnovu biljnih vrsta koje su zastupljene na tom području. Slike su nastale snimanjem terena iz aviona. Slike su tif formarta i velikih su dimenzija i rezolucije. 

Proces obrade započinje učitavanjem tif slike, nakon toga slijedi proces predprocesiranja u kome se velika slika siječe na dimenzije 512x512 piksela. Na ovaj način od jedne velike slike, čija  je veličina i nekoliko megabajta, nastaje više manjih slika nad kojima se vrši predikcija. Predikcija se vrši metodama mašinskog učenja primjenom vgg modela. Ulaz u model je isječena slika, a rezultat je ista slika sa maskama. Model odredi klasu kojoj slika pripada, a na osnovu vraćene maske radi se računanje pocenta zastupljenosti otkrivene biljne vrste. 
Na osnovu dobijenih informacija generiše se ontologija sa individualima nad kojima je moguće raditi SPARQL upite. Upite na ontologijeom je moguće izvršavati iz web aplikacije.
Na slici je prikazano kako treče proces obrade:

# SLIKA arhitektura sistema dodati

#### Skup podataka
Skup podataka se sastoji od svega par slika. Slike su velikih dimenzija i rezolucije. Nastale su snimanjeen određenog močvarnog područja iz aviona. Nad dobijenim slikama je potrebno primjeniti algoritme mašinskog učenja kako bi sa slike dobili informacije o zastupljenosti biljne zajednice na snimanom području. Iz dobijenih informacija se na kraju generiše ontologija, ali o tome svakako više u sekciji ontologija.
Zarad uspješnije obrade, početna slika je sječena na dimenzije 512x512. Tako dobijene slike su anotirane uz pomoć [via-2.0.12](https://www.robots.ox.ac.uk/~vgg/software/via/) alata za anotaciju i tako anotirane slate su modelu na obradu.

#### Model
Za određivanje prisustva biljne zajednice na određenom području, slici, korišten je vgg16 model mašinskog učenja. 

[VGG16](https://arxiv.org/pdf/1409.1556.pdf) je  model mašinskog učenja namjenjen za obradu slike, koji se istakao na takmičenju ImageNet 2014. sa pobjedom u klasifikaciji i detekciji objekata. Ovaj model ima oko 138 miliona parametara i postiže tačnost od 92.7% na 1000 klasa. 
Sa 16 slojeva, VGG16 se često koristi za klasifikaciju slika i lako je primjenjiv i pogodan za dodatna treniranja. Posebna karakteristika ovog modela je korišćenje malih 3x3 filtera u konvolutivnim slojevima, čime se postiglo značajno poboljšanje u odnosu na prethodne arhitekture. Arhitektura VGG16 ima 13 konvolutivnih slojeva, pet slojeva sažimanja (max pooling), i tri potpuno povezana sloja. 

# SLIKA arhitekture modela dodati

Nakon što model izvrši predikciju vrati detektovane klase za proslijeđenu sliku. Na osnovu dobijenih predikcija moguće je vizulizovati detekcije, što je i prikazano na slici. 
# SLIKA detekcije ako izgleda maska dodati


#### Ontologija
Na osnovu dobijene detekcije moguće je odrediti boljnu zajednicu zastupljenu na datoj slici, procent zastupljenosti biljne zajednice kao i kordinatama. Kordinate se mogu izračunati na osnovu pomjeraja kojo je napravljen prilikom sječenja slike. Na osnovu ovih informacija i na osnovu informacija o nazivu slike, položaju u početnoj slici, generišu se ontologija sa individualima.  Kao osnova korištena je [owlready2](https://owlready2.readthedocs.io/en/v0.42/) iz koje su uzete klase na osnovu kojih smo mi izgradili našu ontologiju.
Mi smo koristili sledeće klase:
```Terrene, Image, Empty, Vegetation, Soil, Liquid, TreeCanopy, Water, Rock, Dirt, Waterlilly, Swamp, Grass, Sand```
Kada se prvi put pokrene sistem, generipše se fajl koji čuva ontologiju i definicije individuala, svaki naredi put se novogenerisani individuali dodavaju na kraj fajla.
#### Web app
Kako bi olakšali postavljanja upita nad generisanim individualima, razvijena je web stranica koja uz pomoć jednostavnog grafičkog interfejsa olakšava postavljanje upita. Uz pomoć forme korisnik može postaviti upit. Takođe, omogućeno je slanje slike na predikciju putem ove web stranice.

# SLIKA aplikacije dodati

#### SPARQL upit
SPARQL je upitni jezik koji se koristi za rad s RDF (Resource Description Framework) podacima. RDF je standardni model za razmenu podataka na webu, često korišćen za opisivanje resursa i njihovih svojstava. SPARQL vam omogućava da postavljate upite nad RDF podacima kako biste izvukli informacije koje vas zanimaju.
Ovaj jednostavan upit dozvoljava pretragu individuala samo na osnovu naziva orginalne slike, dok su ostali parametri opcioni i korisnik ih može,a i ne mora unijeti kada šalje upit. U tom slučaju, upit će vratiti sve individuale vezane za proslijđeni naziv slike. Dodatno je moguće mjenjti, klasu kojoj pripada biljna zajednica, kordinate područja u kome radite pretragu, minimalni i maksimalni procenat zastupljenosti te biljne vrste.

### Pokretanje
Da bi ste pokrenuli sistem potrebno je da ispratite slijedeće korake:aplikacije
source ime_venv/bin/activate
```
Nakon aktivacije virtuelnog okruženja, potrebno je instalirati sve potrebne biblioteke koje su navedene u ```requirement.txt``` fajlu. 
```
pip3 install -r requirements.txt
```

Nakon što ispratite par ovih koraka možete započeti sa radom pokretanjem app.py skripte

```
python3 app.py
```

##### Napomene
Ukoliko imate problem sa pokretanjem modela na Vašoj grafičkoj kartici dodajte ovu liniju koda u ```make_prediction.py``` fajl, kako bi ste isključili upotrebu GPU.
```
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
``````