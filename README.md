# AMHE-problem-plecakowy

### Treść zadania:

```
Rozwiazać problem plecakowy dla danych skorelowanych i nieskorelowanych używając algorytmu PBIL (Population-Based Incremental Learning), porównując z wybraną metaheurystyką. Wymagana dokładna analiza statystyczna przedstawionych wyników.
Zalecane konsultacje u prowadzącego przed przystąpieniem do pracy nad projektem.
```

### Literatura:
- DOI:10.3390/app11199136 [link](https://www.mdpi.com/2076-3417/11/19/9136)
- DOI:10.1109/ICNC.2007.126 [link](https://ieeexplore.ieee.org/document/4344579)


### Problem plecakowy

Problem plecakowy (ang. knapsack problem) to klasyczny problem optymalizacyjny, w którym celem jest wybranie spośród dostępnych przedmiotów takich, które zmaksymalizują łączną wartość, nie przekraczając przy tym określonej pojemności plecaka. Każdy przedmiot ma swoją wagę i wartość, a zadanie polega na znalezieniu optymalnego zestawu. Istnieją różne warianty problemu, m.in. problem plecakowy 0/1 (gdzie każdy przedmiot można wziąć tylko raz) oraz plecakowy z podziałem (fractional knapsack), w którym można zabrać ułamek przedmiotu.

### Population-Based Incremental Learning w kontekście problemu plecakowego

1. Inicjalizujemy wektor genomu populacji - np. dla 4 przedmiotów `[0.5, 0.5, 0.5, 0.5]`
2. Generujemy populację na podstawie wektora prawdopodobieństwa.
3. Ewaluujemy i wybieramy najlepszego osobnika.
4. Aktualizujemy wektor genomu zgodnie z wybranym osobnikiem.
5. Kroki powtarzamy do momentu ustabilizowania się wektora genomu populacji.


## Pytania

- Jakie są oczekiwane terminy?
- Pracować na jakichś gotowych danych, czy stworzyć generator? Z jaką ilością przedmiotów działać?
- Czy dobrze zrozumieliśmy algorytm PBIL?
- 'porównując z wybraną metaheurystyką.' - chodzi o porównanie PBIL z klasycznym rozwiązaniem problemu plecakowego, opartym na stosunku wartości do wagi elementu? W jaki sposób miałoby wyglądać takie porównanie? Złożoność czasowa / lepsze rozwiązanie?
- Co ma zawierać się w dokładnej analizie statystycznej?