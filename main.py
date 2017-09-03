# projeto SoluSoja

from skimage import io
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing,disk
import cv2


def main():
    # carregar imagem
    img = io.imread('E:/workspaces/workspacePYCharm/DevAgri/fotos_placas/IMG_4495_2.jpg')

    # converter de RGB para HSV
    imgHsv = color.rgb2hsv(img)

    # copia img original RGB para imgRgb
    imgRgb = np.copy(img)

    # imgRgb é a máscara com a soja na imagem
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # intervalo no espaço de cor HSV em que segmentamos o grão
            if imgHsv[i,j,0] > 0.5:
                if imgHsv[i,j,0] < 0.8:
                    imgRgb[i,j] = 0.0

    # converter para escala de cinza
    imgGray = color.rgb2gray(imgRgb)

    # máscara de fundo
    imgBack = np.zeros((imgGray.shape))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if imgGray[i,j] > 0.0:
                imgBack[i,j] = 1.0

    # binarizar a imagem
    ret, imgBin = cv2.threshold(imgGray, 0.5, 1.0, cv2.THRESH_BINARY)

    # operação de fechamento morfológico para remoção de ruídos na imagem
    imgMorph = closing(imgBin,disk(2))

    # subtração da imagem de fundo pela gerada pela morfologia matemática
    imgSub = imgBack - imgMorph

    # mostrar a área
    area_total = np.sum(imgBack)
    print('Área total: ',area_total)
    area_boa = np.sum(imgMorph)
    print('Área boa: ',area_boa)
    print('Área fermentada: ',area_total-area_boa)
    print('% fermentado: ',((area_total-area_boa)/area_total)*100)

    # impressão das imagens
    titles = ['ORIGINAL','GRAY','SUB','MORPH','FUNDO','BINARIA']
    images = [img,imgGray,imgSub,imgMorph,imgBack,imgBin]

    plt.subplot(2,3,1), plt.imshow(images[0],'gray')
    plt.title(titles[0])
    plt.subplot(2,3,2),plt.imshow(images[1],'gray')
    plt.title(titles[1])
    plt.subplot(2,3,3), plt.imshow(images[2],'gray')
    plt.title(titles[2])
    plt.subplot(2,3,3), plt.imshow(images[2],alpha=0.5)
    plt.subplot(2,3,4), plt.imshow(images[3],'gray')
    plt.title(titles[3])
    plt.subplot(2,3,5), plt.imshow(images[4],'gray')
    plt.title(titles[4])
    plt.subplot(2,3,6), plt.imshow(images[5],'gray')
    plt.title(titles[5])

    plt.show()


main()


