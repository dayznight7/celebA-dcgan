import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dataroot = "./data/celeba/"
    # dataloader worker
    workers = 2
    batch_size = 128
    image_size = 64
    # 이미지 채널수 RGB => 3
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 5
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1

    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    dataset = dsets.ImageFolder(root=dataroot,
                                transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    # 데이터셋 작동 테스트
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()


    # netG와 netD에 적용시킬 커스텀 가중치 초기화 함수
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    class Generator(nn.Module):

        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # 입력 데이터 (RGB)3*64*64
                # 입력데이터 Z가 가장 처음 통과하는 전치 합성곱 계층입니다.
                nn.ConvTranspose2d(in_channels=nz,
                                   out_channels=ngf*8,
                                   kernel_size=(4,4),
                                   stride=(1,1),
                                   padding=0,
                                   bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # 위의 계층을 통과한 데이터의 크기. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # 위의 계층을 통과한 데이터의 크기. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # 위의 계층을 통과한 데이터의 크기. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # 위의 계층을 통과한 데이터의 크기. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # 위의 계층을 통과한 데이터의 크기. (nc) x 64 x 64
            )

        def forward(self, input):
            return self.main(input)


    # 모델 구조 확인
    # netG = Generator(ngpu).to(device)
    # # 필요한 경우 multi-gpu를 설정 해주세요
    # if (device.type == 'cuda') and (ngpu > 1):
    #     netG = nn.DataParallel(netG, list(range(ngpu)))
    # 모든 가중치의 평균을 0, 분산을 0.02로 초기화
    # netG.apply(weights_init)
    # print(netG)


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # 입력 데이터의 크기는 (nc) x 64 x 64 입니다
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 위의 계층을 통과한 데이터의 크기. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 위의 계층을 통과한 데이터의 크기. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 위의 계층을 통과한 데이터의 크기. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # 위의 계층을 통과한 데이터의 크기. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)


    # 모델 구조 확인
    # netD = Discriminator(ngpu).to(device)
    # # 필요한 경우 multi-gpu를 설정 해주세요
    # if (device.type == 'cuda') and (ngpu > 1):
    #     netD = nn.DataParallel(netD, list(range(ngpu)))
    # # 모든 가중치의 평균을 0, 분산을 0.02로 초기화
    # netD.apply(weights_init)
    # print(netD)

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)

    # BCELoss 함수의 인스턴스를 생성합니다
    criterion = nn.BCELoss()

    # 생성자의 학습상태를 확인할 잠재 공간 벡터를 생성합니다
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # 학습에 사용되는 참/거짓의 라벨을 정합니다
    real_label = 1.
    fake_label = 0.

    # G와 D에서 사용할 Adam옵티마이저를 생성합니다
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # 학습 과정

    # 학습상태를 체크하기 위해 손실값들을 저장합니다
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # 에폭(epoch) 반복
    for epoch in range(num_epochs):
        # 한 에폭 내에서 배치 반복
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) D학습 손실함수: log(D(x)) + log(1 - D(G(z)))
            ###########################
            # 진짜 데이터들로 학습
            netD.zero_grad()
            real_cpu = data[0].to(device)
            # data[0] = 이미지 데이터 128*3*8*8
            # data[1] = 라벨 데이터
            b_size = real_cpu.size(0)
            # data: [128(배치사이즈), 3(rgb), 64, 64]
            # 111로 채우기 (3, 64, 64)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # 가짜 데이터들로 학습
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)   # view(-1) : 1차원 텐서로
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # 가짜 이미지와 진짜 이미지 모두에서 구한 손실값들을 더합니다
            # 이때 errD는 역전파에서 사용되지 않고, 이후 학습 상태를 리포팅(reporting)할 때 사용합니다
            errD = errD_real + errD_fake
            # grad는 누적됨
            optimizerD.step()

            ############################
            # (2) G학습 손실함수: log(D(G(z)))를 최대화 합니다
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # 생성자의 손실값을 구하기 위해 진짜 라벨을 이용할 겁니다
            # 우리는 방금 D를 업데이트했기 때문에, D에 다시 가짜 데이터를 통과시킵니다.
            # 이때 G는 업데이트되지 않았지만, D가 업데이트 되었기 때문에 앞선 손실값가 다른 값이 나오게 됩니다
            output = netD(fake).view(-1)
            # G의 손실값을 구합니다
            errG = criterion(output, label)
            # G의 변화도를 계산합니다
            errG.backward()
            D_G_z2 = output.mean().item()
            # G를 업데이트 합니다
            optimizerG.step()

            # 훈련 상태를 출력합니다
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # 이후 그래프를 그리기 위해 손실값들을 저장해둡니다
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # fixed_noise를 통과시킨 G의 출력값을 저장해둡니다
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake, padding=2, normalize=True))

            iters += 1


    torch.save(netG.state_dict(), "G.pth")
    torch.save(netD.state_dict(), "D.pth")

    # 1에폭당 이미지 저장
    # for i, img in enumerate(img_list):
    #     save_image(img, f"generated_images/img_gen{i}.png")


# 참고문헌
# https://tutorials.pytorch.kr/beginner/dcgan_faces_tutorial.html

# dataset download
# https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

