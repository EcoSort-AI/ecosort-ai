# EcoSort AI - Guia de Teste Local (Edge Computing)

Este guia descreve os passos necessários para executar e validar o script de inferência da **Lixeira Inteligente (EcoSort)** num ambiente local. O objetivo é validar a captura de imagem, a classificação via IA (modelo NCNN) e o envio dos dados para a API na Vercel.

## Estrutura de Pastas

Para o teste funcionar, a pasta `edge/` deve estar organizada assim:

```text
edge/
├── main.py              # Script principal
├── requirements.txt     # Dependências (ultralytics, opencv-python-headless, requests, python-dotenv, ncnn)
├── .env                 # Configurações de ambiente
└── best_ncnn_model/     # Pasta do modelo exportado
    ├── model.ncnn.bin
    ├── model.ncnn.param
    └── metadata.yaml
```

## Pré-requisitos

- Python 3.10+ instalado
- Webcam funcional
- Acesso à Internet para comunicar com o backend na Vercel

## Configuração do Ambiente

### 1. Criar e Ativar o Ambiente Virtual (venv)

**No Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**No Linux/Mac:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Configuração do .env

Crie um arquivo .env na raiz da pasta `/edge` de acordo com o conteúdo do .env.example

### Execução

Para iniciar a simulação da lixeira:

```bash
python main.py
```

1. O script carregará o modelo NCNN (otimizado para Edge).
2. Pressione ENTER para simular que um resíduo foi detetado.
3. A IA classificará o objeto e enviará o JSON para a Vercel.
