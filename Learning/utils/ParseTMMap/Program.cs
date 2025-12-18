using System;
using GBX.NET;
using GBX.NET.Engines.Game;
using GBX.NET.LZO;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: ParseTMMap <map.Map.Gbx>");
            return;
        }

        // 🔑 Activation du support LZO
        Gbx.LZO = new Lzo();

        var gbx = Gbx.Parse(args[0]);
        var map = gbx.Node as CGameCtnChallenge;

        if (map == null)
        {
            Console.WriteLine("Erreur : ce fichier n'est pas une map Trackmania.");
            return;
        }

        Console.WriteLine("Block;X;Y;Z;Dir");

        foreach (var block in map.Blocks)
        {
            string blockName = block.BlockModel != null
                ? block.BlockModel.ToString()
                : "UNKNOWN";

            Console.WriteLine(
                $"{blockName};{block.Coord.X};{block.Coord.Y};{block.Coord.Z};{block.Direction}"
            );
        }
    }
}
