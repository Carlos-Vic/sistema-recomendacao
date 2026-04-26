# 📋 GABARITO DE TESTES - SISTEMA DE FALLBACK DE RECOMENDAÇÕES

## Visão Geral

O sistema de recomendação usa uma cascata de fallback com 5 níveis quando não encontra um match perfeito. Este documento serve como referência para testar cada nível.

---

## 🔄 Cascata de Fallback

```
NÍVEL 0: Todas as restrições (família olfativa + ocasião + gênero + preço)
    ↓ (se score < 0.50 ou sem família correta)
NÍVEL 1: Remove ocasião (mantém família + gênero + preço)
    ↓ (se score < 0.50)
NÍVEL 2: Remove gênero (mantém família + ocasião + preço)
    ↓ (se score < 0.50)
NÍVEL 3: Sem restrições - Retorna Top-5 mais vendidos mockados
```

---

## 📊 Casos de Teste Recomendados

### **NÍVEL 0 - Match Perfeito**

| ID | Descrição | Família | Ocasião | Gênero | Preço | Esperado |
|----|-----------|---------|---------|--------|-------|----------|
| 0.1 | Floral clássico | floral | diurno casual | feminino | R$200 | ✅ Nível 0 |
| 0.2 | Amadeirado formal | amadeirado | noturno formal | masculino | R$250 | ✅ Nível 0 |

**Critério de Sucesso:**
- ✅ Score >= 0.50 (threshold mínimo)
- ✅ Pelo menos 1 resultado com a família solicitada

---

### **NÍVEL 1 - Ocasião Restritiva**

| ID | Descrição | Família | Ocasião | Gênero | Preço | Esperado |
|----|-----------|---------|---------|--------|-------|----------|
| 1.2 | Chipré específico | chipre | diurno casual | feminino | R$200 | ⚠️ Nível 1 |

**Quando Ativar:**
- Ocasião muito específica (ex: "diurno verao")
- Nível 0 falha, mas Nível 1 encontra match

**Critério de Sucesso:**
- Busca sem ocasião encontra score >= 0.50
- Produto tem família solicitada

---

### **NÍVEL 2 - Gênero Cruzado**

| ID | Descrição | Família | Ocasião | Gênero | Preço | Esperado |
|----|-----------|---------|---------|--------|-------|----------|
| 2.1 | Aromatico feminino | aromatico | diurno casual | feminino | R$150 | 🔄 Nível 2 |
| 2.2 | Especiado feminino | especiado | diurno casual | feminino | R$100 | 🔄 Nível 2 |

**Quando Ativar:**
- Família é predominantemente masculina
- Nível 1 falha, mas removendo gênero encontra match

**Critério de Sucesso:**
- Busca sem gênero, com ocasião + família
- Encontra score >= 0.50 com família correta

---

### **NÍVEL 3 - Impossível (Top-5 Mockados)**

| ID | Descrição | Família | Ocasião | Gênero | Preço | Esperado |
|----|-----------|---------|---------|--------|-------|----------|
| 3.1 | Combinação extrema | citrico, especiado | noturno inverno | feminino | R$50 | 📌 Nível 3 |
| 3.2 | Preço mínimo com restrições | frutal, amadeirado | noturno inverno | feminino | R$50 | 📌 Nível 3 |

**Quando Ativar:**
- Combinação de critérios é impossível de atender
- Todos os níveis anteriores falham

**Critério de Sucesso:**
- Retorna top-5 mais vendidos para o gênero
- Mensagem indica "Sem match exato"

---

## 🧪 Como Testar

### Passo 1: Rodar o gabarito de lógica
```bash
python gabarito_logica.py
```
Mostra os top-3 resultados em cada nível para entender a cascata.

### Passo 2: Rodar os testes automatizados
```bash
python test_gabarito_fallback.py
```
Valida se todos os 10 casos de teste passam nos níveis esperados.

### Passo 3: Testar manualmente na UI
1. Abra `http://127.0.0.1:7860`
2. Login e vá para "Minha Vitrine"
3. Use os critérios dos casos de teste acima
4. Valide a mensagem de aviso (cor e nível) com a esperada

---

## 📌 Notas Importantes

### ⚠️ Intervalo de Preço Válido
- **Mínimo:** R$50 (limite inferior do slider)
- **Máximo:** R$350 (limite superior do slider)
- Todos os casos de teste usam preços dentro de R$50-R$350

- Sistema verifica se **o primeiro elemento** da lista de famílias está presente no produto
- Exemplo: Se o usuário seleciona `["amadeirado", "oriental"]`, o produto deve ter "amadeirado"

### Threshold de Score
- **Nível 0:** Score >= 0.50 (recomendação de qualidade)
- **Níveis 1-2:** Score >= 0.50 (mesma qualidade)
- **Nível 3:** Sem threshold (retorna mockados)

### Critérios Removidos por Nível
- **Nível 1:** Remove ocasião
- **Nível 2:** Remove gênero
- **Nível 3:** Sem restrições (top-5 mockados)
- Amadeirado feminino: **apenas 1** (Essencial Exclusivo Feminino, R$169.90)
- Chipré feminino: **apenas 2**
- Ocasião "diurno verao": **apenas 2 produtos**

---

## 🔴 Falsos Positivos (Casos Problemáticos)

### Problema: TF-IDF muito permissivo
Um perfume "floral" com notas de sandalo/madeira pode aparecer como match para "amadeirado"

**Solução:** Validação de família olfativa rejeita se:
- Nível 0: Produto não tem a família solicitada
- Recai para próximo nível

### Problema: Limite de ocasiões
Algumas ocasiões são muito raras no catálogo (ex: "diurno verao")

**Resultado:** Nível 1 ativa frequentemente para essas ocasiões

---

## 📈 Métricas de Sucesso

| Métrica | Alvo |
|---------|------|
| Testes Nível 0 passando | 100% |
| Testes Nível 1+ passando | >= 80% |
| Taxa de Nível 4 acionado | < 5% (deve ser raro) |

---

## 💡 Exemplos de Uso Real

### Caso 1: Usuário busca amadeirado feminino com R$50
1. Nível 0: Falha (apenas 1 feminino amadeirado, custa R$169.90)
2. Nível 1: Falha (ainda sem match de preço)
3. Nível 2: Falha (R$57.50 ainda é insuficiente)
4. Nível 3: PASSA (encontra "Luck for Him" masculino amadeirado R$79.90)
5. Resultado: ⚠️ Nível 3 "Poucos perfumes no seu gênero"

### Caso 2: Usuário busca floral feminino diurno casual com R$200
1. Nível 0: PASSA (múltiplos florais femininos existem)
2. Resultado: ✅ Nível 0 "Recomendações personalizadas!"

### Caso 3: Usuário busca aromatico citrico especiado feminino noturno inverno com R$30
1. Nível 0-3: Todos falham (combinação impossível)
2. Nível 4: PASSA (retorna top-5 mockados)
3. Resultado: 📌 Nível 4 "Perfumes mais vendidos"

