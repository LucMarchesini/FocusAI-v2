# Focus Detector — Web App

Versão web do `Modelo_funcional/testar_modelo.py`. Faz exatamente a mesma
detecção de foco em tempo real, mas roda no navegador:

- A webcam é capturada pelo navegador.
- Os frames são enviados ao backend Flask, que roda o **mesmo**
  `Modelo_funcional/focus_model.keras` e devolve a probabilidade.
- O HUD (estado, confiança, threshold, brackets, scan line) é desenhado ao vivo.
- Ao encerrar, é gerado um relatório da sessão (KPIs, donut, produtividade ao
  longo do tempo e linha do tempo dos estados) — equivalente ao dashboard
  matplotlib do desktop.

O modelo treinado **não é duplicado**: o backend o carrega da pasta
`../Modelo_funcional`.

## Como rodar

```bash
cd Web_app
python app.py
```

Abra <http://localhost:5000> no navegador e clique em **Iniciar sessão**
(autorize o acesso à webcam).

> Webcam no navegador exige `localhost` ou HTTPS. Em `localhost` funciona direto.

## Controles

| Ação | Botão | Tecla |
|------|-------|-------|
| Inverter classes | Inverter classes | `I` |
| Aumentar limiar | + limiar | `+` |
| Diminuir limiar | − limiar | `-` |
| Encerrar e ver relatório | Encerrar & relatório | `Q` / `Esc` |

## Dependências

Já presentes no ambiente: `flask`, `keras`, `tensorflow`, `numpy`,
`opencv-python`. Nenhuma biblioteca JS externa é usada (gráficos desenhados em
canvas puro).
