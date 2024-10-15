<div style="display: flex; width: 100%;">
    <div style="flex: 1; padding: 0px;">
        <p>© Albert Palacios Jiménez, 2024</p>
    </div>
    <div style="flex: 1; padding: 0px; text-align: right;">
        <img src="./assets/ieti.png" height="32" alt="Logo de IETI" style="max-height: 32px;">
    </div>
</div>
<br/>

# Exercici 0

Implementa un pla d‘inversió d’un banc, per diferents tipus de clients:

Necessites interfície anomenada ‘Pla’ amb les funcions: 

- calculaImpostos
- calculaInteressos
- getInterès

Crea 3 tipus de plans derivats de ‘Pla’: 

- PlaDomestic
- PlaComercial
- PlaInstitucional

La funció getInterès retorna: 

- 3.5 pel plà domèstic
- 7.4 pel plà comercial
- 5.5 per l’institucional

La funció calculaImpostos rep un valor decimal i retorna:

- valor * getInteres() * 0.21 pel plà domèstic
- valor * getInteres() * 0.15 pel plà comercial
- valor * getInteres() * 0.05 pel plà institucional

La funció calculaInteressos rep un valor decimal, un enter dies i retorna:

- valor * getInteres() * dies / 365

Crea 3 usuaris un de cada tipus, hauràs de fer una classe ‘Usuari’ i extendre’n les altres 3:

- UsuariDomestic
- UsuariComercial
- UsuariInstitucional

La classe usuari té

- Una variable decimal “diners”
- Una cadena “nom”
- Una variable “Pla” que ha d’estar iniciada al tipus d’usuari que és

Les funcions calculaImpostos i calculaInteressos que simplement criden a les del Plà i retornen el valor que correspon

Crea 3 instàncies, una per cada tipus d’usuari

- Fes que l’usuari domèstic tingui 15000 diners
- Fes que l’usuari comercial tingui 350000 diners
- Fes que l’usuari institucional tingui 1000000 diners

Aleshores mostra per pantalla el nom, impostos i interessos de cada usuari, així:

“NomUsuari té X diners, el què significa que paga Y impostos i rep Z interessos” (aplica sempre 365 dies)
