from PIL import Image
import sys
import math
import datetime
import multiprocessing


def read_image(file_name):
    f = Image.open(file_name)
    return f.load()


def write_image(file_name, image_object, resolution):
    image_object.show()
    image_object.save(file_name)


def write_gif(file_name, seq, resolution=[512, 512]):
    img = Image.new('RGB', tuple(resolution))

    img.save(file_name, save_all=True,
             append_images=seq, duration=500, loop=0)
    return img


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
            sum_r += (pixel[i_block * 8 + x, j_block * 8 + y]
                      [0] - 128) * cos_i * cos_j
            sum_g += (pixel[i_block * 8 + x, j_block * 8 + y]
                      [1] - 128) * cos_i * cos_j
            sum_b += (pixel[i_block * 8 + x, j_block * 8 + y]
                      [2] - 128) * cos_i * cos_j
    if i % 8 == 0:
        sum_r /= math.sqrt(2)
        sum_g /= math.sqrt(2)
        sum_b /= math.sqrt(2)
        if j % 8 == 0:
            sum_r /= math.sqrt(2)
            sum_g /= math.sqrt(2)
            sum_b /= math.sqrt(2)

    return [int(sum_r / 4), int(sum_g / 4), int(sum_b / 4)]


def dct_8_block(pixel, start_i, start_j, N=64):
    index = 1
    outp = [[[0, 0, 0]] * 8 for x in range(8)]
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
                    if index > N:
                        break
            else:
                for x in range(lowerBoundx, upperBoundx + 1):
                    row = start_i + x
                    col = start_j + i - x
                    outp[x][i - x] = pixel[row][col]
                    index += 1
                    if index > N:
                        break
        else:
            break
    return outp


def dct_select_coeff(pixel, N=64, resolution=[512, 512]):
    outp = [[] for x in range(resolution[1])]
    for i in range(0, resolution[0], 8):
        for j in range(0, resolution[1], 8):
            block_result = dct_8_block(pixel, i, j, N)
            for k in range(0, 8):
                outp[i + k] += block_result[k]
    return outp


def dct_compress(pixel, resolution=[512, 512]):
    outp = [[(0, 0, 0)] * resolution[0] for x in range(resolution[1])]
    for i in range(0, resolution[0]):
        for j in range(0, resolution[1]):
            outp[i][j] = dct_compress_calc(pixel, i, j)
    return outp


compress_dct_pixels = []
compress_dwt_pixels = []
dct_coeff_pixel = []


def dct_decompress_calc(i, j):

    sum_r = 0
    sum_g = 0
    sum_b = 0
    i_block = i / 8
    j_block = j / 8

    mutli_i = 1
    mutli_j = 1

    for x in range(8):
        if x == 0:
            mutli_i = 1 / math.sqrt(2)
        else:
            mutli_i = 1
        cos_i = mutli_i * math.cos((2 * (i % 8) + 1) * x * math.pi / 16)
        for y in range(8):
            if y == 0:
                mutli_j = 1 / math.sqrt(2)
            else:
                mutli_j = 1
            cos_j = mutli_j * math.cos((2 * (j % 8) + 1) * y * math.pi / 16)
            sum_r += (dct_coeff_pixel[i_block * 8 + x]
                      [j_block * 8 + y][0]) * cos_i * cos_j
            sum_g += (dct_coeff_pixel[i_block * 8 + x]
                      [j_block * 8 + y][1]) * cos_i * cos_j
            sum_b += (dct_coeff_pixel[i_block * 8 + x]
                      [j_block * 8 + y][2]) * cos_i * cos_j

    return tuple([int(sum_r / 4 + 128), int(sum_g / 4 + 128), int(sum_b / 4 + 128)])


def dct_decompress(N):
    global dct_coeff_pixel
    resolution = [512, 512]
    img = Image.new('RGB', tuple(resolution))
    outp = img.load()
    dct_coeff_pixel = dct_select_coeff(compress_dct_pixels, N)
    for i in range(0, resolution[0]):
        for j in range(0, resolution[1]):
            outp[i, j] = dct_decompress_calc(i, j)
    return img


def main():
    # YourProgram.exe C:/myDir/myImage.rgb 1234
    global compress_dct_pixels
    global compress_dwt_pixels
    now = datetime.datetime.now()
    image_file_name = sys.argv[1]
    coeff = int(sys.argv[2])

    # Read file
    input_pixels = read_image(image_file_name)
    # DCT compress
    #compress_dct_pixels = dct_compress(input_pixels)
    # DWT compress
    compress_dwt_pixels = dwt_compress(input_pixels)
    if coeff == -1:
        #input_seq = []
        process_workers = multiprocessing.Pool(10)
        """
        for n in range(1, 3):
            #output_pixels = dct_decompress(dct_pixels, N=n)
            input_seq.append([dct_pixels,n])
            #write_image('result{}.bmp'.format(str(n)), output_pixels, [512, 512])
            print datetime.datetime.now() - now
        """
        write_gif("result.gif", process_workers.map(
            dct_decompress, list(range(1, 65))))
        process_workers.close()
        print "Animation (name: 'result.gif') saved"
    else:
        block_8_coeff = round(coeff / 4096)
        # DCT 
        #output_dct_pixels = dct_decompress(N=block_8_coeff)
        # DWT
        output_dwt_pixels = dwt_decompress(N=coeff)
        write_image('result.bmp', output_dwt_pixels, [512, 512])

        print "Image (name: 'result.bmp') saved"
    print datetime.datetime.now() - now
    # write_image('original1.bmp', input_pixels, resolution)


# DWT implementation
def dwt_compress(pixels, resolution=[512, 512]):

    output_pixels = [[0] * resolution[0] for x in range(resolution[1])]

    for i in range(resolution[0]):
        for j in range(resolution[1]):
            output_pixels[i][j] = pixels[i, j]

    return dwt_rec_row(output_pixels, resolution[0] / 2, resolution[1])

def dwt_decompress(N=262144):
    resolution = [512, 512]
    img = Image.new('RGB', tuple(resolution))
    outp = img.load()
    output_pixels = dwt_coeff_pixel(compress_dwt_pixels, N)

    for i in range(0, resolution[0]):
        for j in range(0, resolution[1]):
            outp[i, j] = tuple([int(x) for x in output_pixels[i][j]])
    return img


def dwt_coeff_pixel(pixel, N=262144):
    index = 1
    outp = [[[0, 0, 0]] * 512 for x in range(512)]
    for i in range(262143):
        upperBoundx = min(i, 511)
        lowerBoundx = max(0, i - 511)
        upperBoundy = min(i, 511)
        lowerBoundy = max(0, i - 511)
        if index <= N:
            if i % 2 == 0:
                for y in range(lowerBoundy, upperBoundy+1):
                    outp[i - y][y] = pixel[i-y][y]
                    index += 1
                    if index > N:
                        break
            else:
                for x in range(lowerBoundx, upperBoundx+1):
                    outp[x][i - x] = pixel[x][i-x]
                    index += 1
                    if index > N:
                        break
        else:
            break
    return outp


def dwt_rec_row(pixels, row_length, col_length):

    if row_length == 1:
        return pixels

    output_pixels = pixels[0:]

    for i in range(col_length):
        for j in range(row_length):
            output_pixels[i][j] = [(pixels[i][j * 2][x] + pixels[i][j * 2 + 1][x]) / 2.0 for x in range(3)]
            output_pixels[i][row_length +j] = [(pixels[i][j * 2][x] - pixels[i][j * 2 + 1][x]) / 2.0 for x in range(3)]

    return dwt_rec_col(output_pixels, row_length, col_length/2)


def dwt_rec_col(pixels, row_length, col_length):

    if col_length == 1:
        return pixels

    output_pixels = pixels[0:]

    for i in range(row_length):
        for j in range(col_length):
            output_pixels[j][i] = [(pixels[j * 2][i][x] + pixels[j * 2 + 1][i][x]) / 2.0 for x in range(3)]
            output_pixels[col_length + j][i] = [(pixels[j * 2][i][x] - pixels[j * 2 + 1][i][x]) / 2.0 for x in range(3)]

    return dwt_rec_row(output_pixels, row_length/2, col_length)



main()