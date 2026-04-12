#!/bin/bash
# Launch g6e.12xlarge for Multilingual 1B v3 training
# 4× L40S GPUs (48GB each), 48 vCPUs, 384GB RAM
# Cost: ~$4.65/hr on-demand, training ~6.5h = ~$30

set -e

S3_BUCKET="autoresearch-dashboard-196766918360"

# Upload training script to S3
aws s3 cp /home/ubuntu/.openclaw/workspace/multilingual-7b/train_multilingual_1b_v3.py \
  s3://$S3_BUCKET/multilingual-7b/1b-model/train_multilingual_1b_v3.py

echo "Training script uploaded to S3"

# User data script for GPU instance
cat > /tmp/v3_userdata.sh << 'USERDATA'
#!/bin/bash
set -ex

# Setup
export DEBIAN_FRONTEND=noninteractive
cd /tmp

# Install Python packages
pip3 install torch sentencepiece numpy boto3 tqdm 2>/dev/null || \
  pip install torch sentencepiece numpy boto3 tqdm

S3_BUCKET="autoresearch-dashboard-196766918360"

# Download training data (v2 = expanded Arabic)
mkdir -p /tmp/training-data
aws s3 sync s3://$S3_BUCKET/multilingual-7b/training-data-v2/ /tmp/training-data/

# Download tokenizer
aws s3 cp s3://$S3_BUCKET/multilingual-7b/tokenizer/multilingual_32k.model /tmp/tokenizer.model

# Download training script
aws s3 cp s3://$S3_BUCKET/multilingual-7b/1b-model/train_multilingual_1b_v3.py /tmp/train.py

# Wait for GPUs
for i in $(seq 1 30); do
  nvidia-smi && break
  echo "Waiting for GPUs... ($i)"
  sleep 10
done

# Log GPU info
nvidia-smi
echo "GPUs ready, starting training..."

# Launch DDP training on 4 GPUs
torchrun --nproc_per_node=4 --master_port=29500 /tmp/train.py

echo "=== TRAINING COMPLETE ==="

# Upload all results
aws s3 sync /tmp/checkpoints/ s3://$S3_BUCKET/multilingual-7b/checkpoints/v3/
aws s3 cp /tmp/training.log s3://$S3_BUCKET/multilingual-7b/checkpoints/v3_training.log
aws s3 cp /tmp/eval_results.json s3://$S3_BUCKET/multilingual-7b/checkpoints/v3_eval_results.json

echo "=== ALL UPLOADED TO S3 ==="
USERDATA

echo "User data script created"

# Find latest Deep Learning AMI
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 22.04*" \
            "Name=architecture,Values=x86_64" \
            "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text)

echo "AMI: $AMI_ID"

# Get the central-instances SG (has SSH)
SG_ID="sg-00bcfb47f10b8923e"

# Get first public subnet
SUBNET_ID=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=vpc-0efcd6bcf469e08f6" \
            "Name=tag:Name,Values=*public*" \
  --query 'Subnets[0].SubnetId' --output text 2>/dev/null)

# Fallback to any subnet with IGW route
if [ "$SUBNET_ID" = "None" ] || [ -z "$SUBNET_ID" ]; then
  SUBNET_ID=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=vpc-0efcd6bcf469e08f6" \
    --query 'Subnets[0].SubnetId' --output text)
fi

echo "Subnet: $SUBNET_ID"

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type g6e.12xlarge \
  --key-name "" \
  --iam-instance-profile Name=central-admin-profile \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SG_ID" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","Encrypted":true}}]' \
  --user-data file:///tmp/v3_userdata.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=multilingual-1b-v3-training},{Key=Project,Value=multilingual-7b},{Key=Environment,Value=experiment},{Key=Owner,Value=loki}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "=== Instance launched: $INSTANCE_ID ==="
echo "Type: g6e.12xlarge (4× L40S, 48GB each)"
echo "Cost: ~$4.65/hr"
echo "Expected training time: ~6.5h"
echo "Expected total cost: ~$30"
echo ""
echo "Monitor: aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name'"
echo "Logs will upload to s3://$S3_BUCKET/multilingual-7b/checkpoints/v3_training.log"
