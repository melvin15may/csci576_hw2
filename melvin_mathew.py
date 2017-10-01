from PIL import Image
import sys
import math
import os
import datetime

def read_image(file_name):
    f = Image.open(file_name)
    return f.load()


def write_image(file_name, pixels, resolution):
    img = Image.new('RGB', tuple(resolution))
    outp = img.load()

    for i in range(0, resolution[0]):
        for j in range(0, resolution[1]):
            # print pixels[i,j]
            outp[i, j] = pixels[i, j]

    #img.show()
    img.save(file_name)

def write_gif(file_name, seq, resolution=[512,512]):
	img = Image.new('RGB',tuple(resolution))

	img.save(file_name, save_all=True, append_images=seq, duration=2000, loop=0)
	return


def dct_compress_calc(pixel, i, j):

    sum_r = 0
    sum_g = 0
    sum_b = 0
    i_block = i / 8
    j_block = j / 8
    for x in range(8):
        cos_i = math.cos((2 * x + 1) * (i % 8) * math.pi / 16)
        for y in range(8):
            cos_j = math.cos((2 * y + 1) * (j % 8) * math.pi / 16)
            sum_r += (pixel[i_block * 8 + x, j_block * 8 + y][0] - 128) * cos_i * cos_j
            sum_g += (pixel[i_block * 8 + x, j_block * 8 + y][1] - 128) * cos_i * cos_j
            sum_b += (pixel[i_block * 8 + x, j_block * 8 + y][2] - 128) * cos_i * cos_j
    if i%8 == 0:
    	sum_r /= math.sqrt(2)
    	sum_g /= math.sqrt(2)
    	sum_b /= math.sqrt(2)
   	if j%8 == 0:
   		sum_r /= math.sqrt(2)
   		sum_g /= math.sqrt(2)
   		sum_b /= math.sqrt(2)

    return [int(sum_r/4), int(sum_g/4), int(sum_b/4)]


def dct_8_block(pixel, start_i, start_j, N=64):
    index = 1
    outp = [[[0,0,0]] * 8 for x in range(8)]
    for i in range(15):
        upperBoundx = min(i, 7)
        lowerBoundx = max(0, i - 7)
        upperBoundy = min(i, 7)
        lowerBoundy = max(0, i - 7)
        if index <= N:
            if i % 2 == 0:
                for y in range(lowerBoundy, upperBoundy + 1):
                    row = start_i + i - y
                    col = start_j + y
                    outp[i - y][y] = pixel[row][col]
                    index += 1
            else:
                for x in range(lowerBoundx, upperBoundx + 1):
                    row = start_i + x
                    col = start_j + i - x
                    outp[x][i - x] = pixel[row][col]
                    index += 1
        else:
        	break
    return outp


def dct_select_coeff(pixel, N=64, resolution=[512,512]):
	outp = [[] for x in range(resolution[1])]
	#print outp
	for i in range(0,resolution[0], 8):
		for j in range(0,resolution[1], 8):
			block_result = dct_8_block(pixel, i, j, N)
			for k in range(0,8):
				outp[i+k] += block_result[k]
	return outp


def dct_compress(pixel, resolution=[512, 512]):
	outp = [[(0,0,0)] * resolution[0] for x in range(resolution[1])]
	for i in range(0, resolution[0]):
		for j in range(0, resolution[1]):
			outp[i][j] = dct_compress_calc(pixel,i,j)
	return outp
	#outp = [[(0,0,0)] for x in range(resolution[1])]
	#print outp
	"""
    for i in range(0,resolution[0], 8):
        for j in range(0,resolution[1], 8):
            block_result = dct_8_block(pixel, i, j,N)
            for k in range(0,8):
            	outp[i+k] += block_result[k]
    return outp
	"""


def dct_decompress_calc(pixel, i, j):

    sum_r = 0
    sum_g = 0
    sum_b = 0
    i_block = i / 8
    j_block = j / 8
    
    mutli_i = 1
    mutli_j = 1
    
    for x in range(8):
    	if x ==0:
    		mutli_i = 1/math.sqrt(2)
    	else:
    		mutli_i = 1 
        cos_i = mutli_i * math.cos((2 * (i % 8) + 1) * x * math.pi / 16)
        for y in range(8):
        	if y ==0:
        		mutli_j = 1/math.sqrt(2)
        	else:
        		mutli_j = 1 
        	cos_j = mutli_j * math.cos((2 * (j % 8) + 1) * y * math.pi / 16)
        	sum_r += (pixel[i_block * 8 + x][j_block * 8 + y][0]) * cos_i * cos_j
        	sum_g += (pixel[i_block * 8 + x][j_block * 8 + y][1]) * cos_i * cos_j
        	sum_b += (pixel[i_block * 8 + x][j_block * 8 + y][2]) * cos_i * cos_j

    return tuple([int(sum_r/4 + 128), int(sum_g/4 + 128), int(sum_b/4 + 128)])


def dct_decompress(pixel, resolution=[512, 512]):
    img = Image.new('RGB', tuple(resolution))
    outp = img.load()
    for i in range(0, resolution[0]):
        for j in range(0, resolution[1]):
            outp[i, j] = dct_decompress_calc(pixel,i,j)
    return img


def main():
    # YourProgram.exe C:/myDir/myImage.rgb 1234

    now = datetime.datetime.now()
    image_file_name = sys.argv[1]
    coeff = int(sys.argv[2])

    # Read file
    input_pixels = read_image(image_file_name)

    if coeff == -1:
    	dct_pixels = dct_compress(input_pixels)
    	seq = []
    	for n in range(1,3):
    		output_pixels = dct_decompress(dct_select_coeff(dct_pixels, N=n))
    		seq.append(output_pixels)
    		#write_image('result{}.bmp'.format(str(n)), output_pixels, [512, 512])
    		print datetime.datetime.now() - now
    	write_gif("result.gif",seq)
    else:
    	block_8_coeff = round(coeff / 4096)
    	output_pixels = dct_decompress(dct_select_coeff(dct_compress(input_pixels), N=block_8_coeff))
    	write_image('result.bmp', output_pixels, [512, 512])
    
    print "Image (name: 'result.bmp') saved"
    print datetime.datetime.now() - now
    # write_image('original1.bmp', input_pixels, resolution)

main()
