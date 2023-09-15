from xml.etree import ElementTree as ET
import os
import argparse
import tqdm


def convert_XML_to_DOTA(filename, dst, silence=False):
    mydoc = ET.parse(filename)
    root = mydoc.getroot()

    objects = root.find('objects')
    items = objects.findall('object')
    output_file = os.path.splitext(os.path.split(filename)[-1])[0] + '.txt'
    with open(os.path.join(dst, output_file), 'w') as f:
        ann_list = []
        source = root.find('source')
        source_name = source.find('origin').text
        f.write(f'imagesource:{source_name}\ngsd:1.0\n')
        for item in items:
            label = item.find('possibleresult')
            points = item.find('points')
            label = label.find('name').text
            new_label = label.lower().replace(' ', '-')
            points = [[int(float(item)) for item in point.text.split(',')] for point in points.findall('point')]
            x1, y1 = points[0]
            x2, y2 = points[1]
            x3, y3 = points[2]
            x4, y4 = points[3]
            ann = [x1, y1, x2, y2, x3, y3, x4, y4, new_label, 1]
            ann = [str(item) for item in ann]
            ann_list.append(' '.join(ann))
            if not silence:
                print(new_label, x1, y1, x2, y2, x3, y3, x4, y4)

        f.write('\n'.join(ann_list))

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('datafolder', help='train config file path')
    parser.add_argument('--silence', action='store_true', default=False, help='train config file path')
    args = parser.parse_args()
    label_sets = [
        'train',
        'validation'
    ]
    for set in label_sets:
        src_folder = os.path.join(args.datafolder, set, 'labelXml')
        dota_folder = os.path.join(args.datafolder, set, 'labelTxt')
        xml_files = os.listdir(src_folder)
        os.makedirs(dota_folder, exist_ok=True)
        for file in tqdm.tqdm(xml_files):
            convert_XML_to_DOTA(os.path.join(src_folder, file), dota_folder, args.silence)