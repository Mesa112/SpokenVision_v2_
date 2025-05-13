#!/bin/bash

VERSION=v5 # change version as needed

echo "🔨 Building Docker image..."
docker build --platform=linux/amd64 -t us-central1-docker.pkg.dev/spokenvision/spokenvision-repo/spokenvision:$VERSION .

echo "📤 Pushing Docker image..."
docker push us-central1-docker.pkg.dev/spokenvision/spokenvision-repo/spokenvision:$VERSION

echo "🚀 Deploying to Cloud Run..."
gcloud run deploy spokenvision \
  --image=us-central1-docker.pkg.dev/spokenvision/spokenvision-repo/spokenvision:$VERSION \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated \
  --vpc-connector=spokenvision-connector \
  --vpc-egress=all-traffic \
  --memory=16Gi \
  --cpu=4 \
  --concurrency=1 \
  --timeout=900
