# EcoSort AI - Documentação (Edge Computing)

Este guia descreve os passos necessários para executar e validar o script de inferência da **Lixeira Inteligente (EcoSort)**. O objetivo é validar a captura de imagem, a classificação via IA (modelo NCNN) e o envio dos dados para a API na Vercel.

## Estrutura de Pastas

Para o teste funcionar, a pasta `edge/` deve estar organizada assim:

```text
edge/
├── main.py              # Script principal
├── requirements.txt     # Dependências (ultralytics, opencv-python-headless, requests, python-dotenv, ncnn)
├── Dockerfile           # Instruções para a criação da imagem
├── docker-compose.yml   # Arquivo para definir e orquestrar a aplicação dos containers
├── .env                 # Configurações de ambiente
├── .dockerignore        # Arquivos que o docker deve ignorar ao construir uma imagem
└── best_ncnn_model/     # Pasta do modelo exportado
    ├── model.ncnn.bin
    ├── model.ncnn.param
    └── metadata.yaml
```

## Pré-requisitos

- Docker instalado
- Câmera USB ou módulo de câmera conectado (verifique com ls /dev/video*, deve retornar /dev/video0)
- Acesso à Internet para comunicar com o backend na Vercel

## Configuração do Ambiente

### 1. Acesso ao Raspberry Pi 5 via SSH

```bash
ssh smart-bin-XX@<IP-DO-RASPBERRY>
```
### 2. Navegar até a pasta do projeto

```bash
cd ~/ecosort-ai/edge
```

### 3. Configuração do .env

Crie um arquivo .env na raiz da pasta `/edge` de acordo com o conteúdo do .env.example

```bash
cp .env.example .env
```
```bash
nano .env
```

## Configurar o Docker

Verificar se o Docker está instalado na máquina. Caso não esteja, siga os passos a seguir:

### Instalar o Docker

```bash
curl -sSL https://get.docker.com | sh
```
### Dar permissões ao seu utilizador

```bash
sudo usermod -aG docker $USER
```

### Aplicar as permissões

```bash
newgrp docker
```

## Execução via Docker Compose

Inicie os serviços (a aplicação EcoSort e o Watchtower para atualizações automáticas) em segundo plano:

```bash
docker compose up -d
```
Para acompanhar os logs em tempo real e verificar se a câmera iniciou e o modelo carregou corretamente:

```bash
docker compose logs -f ecosort-edge
```
Verificar conexão com a câmera

```bash
ls /dev/video*
```
Deve retornar /dev/video0 ou algo parecido
## Iniciar as Detecções de Lixo

Para futuras atualizações, a lixeira deve identificar automaticamente a presença de resíduos por meio de sensores. Enquanto isso, execute o passo a passo a seguir para iniciar as detecções:

A aplicação atual funciona em Modo Headless. Em vez de exigir que você pressione a tecla "ENTER" em um terminal interativo de tela, a IA aguarda um sinal virtual em formato de arquivo para classificar o resíduo e enviar os dados para a Vercel.

Para simular que um resíduo foi detectado pela lixeira, rode o seguinte comando no terminal SSH:

```bash
docker exec ecosort-edge touch trigger.txt
```

- O comando cria temporariamente o arquivo `trigger.txt` dentro do container.
- O script `main.py` reconhece o arquivo, deleta-o imediatamente para evitar múltiplas leituras, e captura o frame atual da câmera.
- O modelo YOLO/NCNN classifica o objeto na imagem.
- O resultado (nome da classe e confiança) é exibido nos logs e enviado em formato JSON para a API na Vercel.

## Atualizações Automáticas

O `docker-compose.yml` inclui o Watchtower. Ele verifica automaticamente o Docker Hub a cada 60 segundos procurando por novas versões da imagem `pdec5504/ecosort-edge:latest`. Ao fazer um push de uma nova imagem, a lixeira inteligente será atualizada e reiniciada sozinha, sem necessidade de intervenção manual.
