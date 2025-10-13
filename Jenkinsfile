pipeline {
    agent {
        kubernetes {
            label 'jenkins-agent'
            defaultContainer 'jnlp'
            yamlFile 'k8s/pod-template.yml'
        }
    }

    environment {
        PROJECT_ID = 'road-for-terraform'
        REGION = 'us-east1'
        REPO_NAME = 'mlops-repo'
        IMAGE_NAME = 'ml-inference'
        CLUSTER_NAME = 'dev-instance'
        MODEL_PATH_1 = 'models/eth_price_predictor.pkl'
        MODEL_PATH_2 = 'models/ethusdt_price_predictor.pkl'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Configurar GCP') {
            steps {
                container('gcp-tools') {
                    sh '''
                    gcloud config set project $PROJECT_ID
                    gcloud auth configure-docker $REGION-docker.pkg.dev -q
                    '''
                }
            }
        }

        stage('Instalar Dependências Python') {
            steps {
                container('python') {
                    sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }

        stage('Preparar e Treinar Modelos') {
            steps {
                container('python') {
                    sh '''
                    . venv/bin/activate

                    export CI=true

                    python main.py
                    '''
                }
            }
        }

        stage('Versionamento dos Modelos') {
            steps {
                container('gcp-tools') {
                    sh '''
                    gsutil cp $MODEL_PATH_1 gs://$PROJECT_ID-model-registry/eth-model-$(date +%Y%m%d%H%M).pkl
                    gsutil cp $MODEL_PATH_2 gs://$PROJECT_ID-model-registry/ethusdt-model-$(date +%Y%m%d%H%M).pkl
                    '''
                }
            }
        }

        stage('Build e Push da Imagem') {
            steps {
                container('gcp-tools') {
                    sh '''
                    # Garantir que os models treinados sejam incluídos no build context
                    echo "📦 Verificando models antes do build:"
                    ls -lah models/
                    
                    # Criar cloudbuild.yaml temporário para garantir que os models sejam incluídos
                    cat > cloudbuild.yaml <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - '$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$BUILD_NUMBER'
      - '--no-cache'
      - '.'
images:
  - '$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$BUILD_NUMBER'
timeout: '1800s'
EOF
                    
                    set +e
                    gcloud builds submit \
                      --config=cloudbuild.yaml \
                      --project=$PROJECT_ID \
                      . 2>&1 | tee build_output.txt
                    
                    BUILD_ID=$(grep -i "ID:" build_output.txt | head -1 | sed 's/.*ID: *//i' | awk '{print $1}')
                    
                    if [ -z "$BUILD_ID" ]; then
                      echo "Não foi possível obter o Build ID, mas verificando imagem..."
                      sleep 30
                      if gcloud artifacts docker images describe $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$BUILD_NUMBER --project=$PROJECT_ID 2>/dev/null; then
                        echo "Imagem criada com sucesso!"
                        exit 0
                      else
                        echo "Falha ao criar imagem"
                        exit 1
                      fi
                    fi
                    
                    echo "Verificando status do build: $BUILD_ID"
                    STATUS=$(gcloud builds describe $BUILD_ID --project=$PROJECT_ID --format="value(status)" 2>/dev/null || echo "UNKNOWN")
                    
                    while [ "$STATUS" = "WORKING" ] || [ "$STATUS" = "QUEUED" ]; do
                      echo "Build ainda em andamento... aguardando 10s"
                      sleep 10
                      STATUS=$(gcloud builds describe $BUILD_ID --project=$PROJECT_ID --format="value(status)")
                    done
                    
                    echo "Build Status: $STATUS"
                    
                    if [ "$STATUS" = "SUCCESS" ]; then
                      echo "Build concluído com sucesso!"
                      exit 0
                    else
                      echo "Build falhou com status: $STATUS"
                      exit 1
                    fi
                    '''
                }
            }
        }

        stage('Deploy no GKE') {
            steps {
                container('gcp-tools') {
                    sh '''
                    # Verificar se kubectl está disponível
                    if ! command -v kubectl &> /dev/null; then
                        echo "❌ kubectl não encontrado. Instalando..."
                        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
                        chmod +x kubectl
                        sudo mv kubectl /usr/local/bin/ 2>/dev/null || mv kubectl /usr/bin/
                    else
                        echo " kubectl já instalado: $(which kubectl)"
                    fi
                    
                    # Verificar se gke-gcloud-auth-plugin está disponível
                    if ! command -v gke-gcloud-auth-plugin &> /dev/null; then
                        echo " gke-gcloud-auth-plugin não encontrado. Instalando..."
                        PLUGIN_URL="https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/gke-gcloud-auth-plugin"
                        curl -LO "$PLUGIN_URL" 2>/dev/null || echo "URL alternativo..."
                        
                        # Tentar instalar via apt se disponível
                        if command -v apt-get &> /dev/null; then
                            apt-get update && apt-get install -y google-cloud-sdk-gke-gcloud-auth-plugin
                        else
                            echo "  Não foi possível instalar gke-gcloud-auth-plugin automaticamente"
                        fi
                    else
                        echo " gke-gcloud-auth-plugin já instalado: $(which gke-gcloud-auth-plugin)"
                    fi
                    
                    echo ""
                    echo " Verificando instalações..."
                    kubectl version --client --short 2>/dev/null || kubectl version --client
                    gcloud version | head -n 1
                    
                    export USE_GKE_GCLOUD_AUTH_PLUGIN=True
                    
                    gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID
                    
                    kubectl apply -f k8s/python-namespace.yml
                    
                    export IMAGE_TAG=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$BUILD_NUMBER
                    sed -i "s|REGISTRY/REPO/ml-inference:latest|$IMAGE_TAG|g" k8s/python-deployment.yml
                    
                    kubectl apply -f k8s/python-deployment.yml
                    kubectl apply -f k8s/python-service.yml
                    kubectl apply -f k8s/python-service-monitor.yml
                    
                    kubectl rollout status deployment/ml-inference -n ml-inference --timeout=5m
                    
                    echo "=== Pods ==="
                    kubectl get pods -n ml-inference
                    echo "=== Services ==="
                    kubectl get svc -n ml-inference
                    echo "=== ServiceMonitor ==="
                    kubectl get servicemonitor -n ml-inference
                    '''
                }
            }
        }
    }

    post {
        always {
            script {
                try {
                    cleanWs(
                        deleteDirs: true,
                        disableDeferredWipeout: true,
                        notFailBuild: true
                    )
                } catch (Exception e) {
                    echo "Erro ao limpar workspace: ${e.message}"
                }
            }
        }
    }
}